#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void decideOutputDataType(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateOutputQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model);

void updateOutputQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    //NOTE: This pass will generate output Quantization Params when they are not defined...
    //Here we search for the minimum, maximum possible solution (better range) for the output Activation Tensor
    //Input(Imin,Imax)     Weights(Wmin,Wmax)
    // \                   /
    //  \                 /
    //   \               /
    //    \             /
    //     \           /
    //      \         /
    //       \       /
    //          Conv
    //           |
    //       Output(Omin,Omax)
    //           |
    //        Bias(Bmin,Bmax)
    // Suggestion: Omin = Imin * Wmin * kernel_w * kernel_h * input_channels, Rmin = Omin + Bmin
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::vector<std::string> convolution_types = {"Conv", "DepthwiseConv", "ChannelMajorConvolution"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfConvolution = om.getOpsOfTypes(convolution_types);
    std::vector <mv::Data::OpListIterator> convolutions = {};
    convolutions.reserve(operationsOfConvolution["Conv"].size() + operationsOfConvolution["Depthwise"].size() + operationsOfConvolution["ChannelMajorConvolution"].size());
    convolutions.insert(convolutions.end(), operationsOfConvolution["Conv"].begin(), operationsOfConvolution["Conv"].end());
    double inf = std::numeric_limits<double>::infinity();
    auto maxPoolOps = om.getOps("MaxPool");
    for(auto& opIt : maxPoolOps)
    {
        auto output = opIt->getOutputTensor(0);
        auto input = opIt->getInputTensor(0);

        if (!output->hasAttr("quantParams")
                || output->get<mv::QuantizationParams>("quantParams").isNeutral())
        {
            if (!input->hasAttr("quantParams"))
            {
                if (input->get<mv::QuantizationParams>("quantParams").isNeutral())
                    continue;
            }
            else
            {
                auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");

                output->set<mv::QuantizationParams>("quantParams", inputQuantization);
                opIt->set<mv::QuantizationParams>("quantParams", inputQuantization);
            }
        }

    }

    // Find ScaleShifts (converted to Depthwise in replacement pass)
    std::vector <mv::Data::OpListIterator> scaleshifts = {};
    scaleshifts.reserve(operationsOfConvolution["DepthwiseConv"].size());
    for (auto opIt=operationsOfConvolution["DepthwiseConv"].begin(); opIt != operationsOfConvolution["DepthwiseConv"].end(); ++opIt)
    {
        if ((*opIt)->hasAttr("isScaleShift") && (*opIt)->get<bool>("isScaleShift"))
            scaleshifts.push_back(*opIt);
    }

    for(auto& opIt : scaleshifts)
    {
        auto output = opIt->getOutputTensor(0);
        auto input = opIt->getInputTensor(0);
        auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

        if (!output->hasAttr("quantParams")
                || output->get<mv::QuantizationParams>("quantParams").isNeutral())
        {
            double outWeightsMin = inf;
            double outWeightsMax = -inf;
            double outBiasesMin = inf;
            double outBiasesMax = -inf;

            auto& newInputQuantization = input->get<mv::QuantizationParams>("quantParams");
            auto weights = opIt->getInputTensor("weights");
            auto kernelShape = weights->getShape();
            auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
            auto weights_scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
            auto weights_zp = extendToK(outputChannels, weightsQuantization.getZeroPoint(), weights->getName());

            std::vector<double> outScale(1);
            std::vector<int64_t> outZp(1);
            // input range
            auto minIn = newInputQuantization.getMin()[0];
            auto maxIn = newInputQuantization.getMax()[0];

            bool hasBias = opIt->hasAttr("bias");
            mv::Data::TensorIterator bias;
            if (hasBias)
            {
                bias = dm.getTensor(opIt->get<std::string>("bias"));
            }
            double_t real_weight, real_bias;
            for (size_t c = 0; c < kernelShape[mv::IO_CHANNEL_DIMENSION]; c++)
            {
                double biasScale = weights_scale[c] * newInputQuantization.getScale()[0];

                auto currWeight = (int64_t)weights->at(c);
                real_weight = ((int64_t) currWeight - weights_zp[c]) * weights_scale[c];
                // weights range
                if (real_weight < outWeightsMin)
                    outWeightsMin = real_weight;
                if (real_weight > outWeightsMax)
                    outWeightsMax = real_weight;

                if (hasBias)
                {
                    // biases range
                    real_bias = ((double) bias->at(c)) * biasScale;
                    if (real_bias < outBiasesMin)
                        outBiasesMin = real_bias;
                    if (real_bias > outBiasesMax)
                        outBiasesMax = real_bias;
                }
            }
            // Calculate outputs range
            double outputMin = (minIn * outWeightsMin) + outBiasesMin;
            double outputMax = (maxIn * outWeightsMax) + outBiasesMax;

            calcZeroPointAndScalePerTensor(outputMax, outputMin, outScale[0], outZp[0]);

            mv::QuantizationParams newOutputQuantization = {outZp,outScale,{outputMin},{outputMax}};
            output->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
            opIt->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
        }
    }

    for(auto& opIt : convolutions)
    {
        auto output = opIt->getOutputTensor(0);
        auto input = opIt->getInputTensor(0);
        auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

        if (!output->hasAttr("quantParams")
                || output->get<mv::QuantizationParams>("quantParams").isNeutral())
        {
            double outputMin = inf;
            double outputMax = -inf;

            std::vector<double> outMin(outputChannels, inf);
            std::vector<double> outMax(outputChannels, -inf);

            //Note: if input Tensor has min, max of infs...we need to compute them
            updateInfMinMaxPerTensor(input);

            auto& newInputQuantization = input->get<mv::QuantizationParams>("quantParams");
            auto weights = opIt->getInputTensor("weights");
            auto kernelShape = weights->getShape();
            auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
            auto weights_scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
            auto weights_zp = extendToK(outputChannels, weightsQuantization.getZeroPoint(), weights->getName());

            //input/output quantization are per tensor, weights, bias quantization are per channel
            std::vector<double> outScale(1);
            std::vector<int64_t> outZp(1);
            auto minIn = newInputQuantization.getMin();
            auto maxIn = newInputQuantization.getMax();

            bool hasBias = opIt->hasAttr("bias");
            mv::Data::TensorIterator bias;
            if (hasBias)
            {
                bias = dm.getTensor(opIt->get<std::string>("bias"));
            }
            double_t real_weight, real_bias;
            for (size_t k = 0; k < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; k++)
            {
                double sum_weight = 0;
                double outputMinC = 0;
                double outputMaxC = 0;
                double biasScale = weights_scale[k] * newInputQuantization.getScale()[0];

                for (size_t c = 0; c < kernelShape[mv::KERNEL_INPUT_CHANNELS]; c++)
                    for (size_t h = 0; h < kernelShape[mv::KERNEL_HEIGHT]; h++)
                        for (size_t w = 0; w < kernelShape[mv::KERNEL_WIDTH]; w++)
                        {
                            auto currWeight = (int64_t)weights->at({w,h,c,k});
                            real_weight = ((int64_t) currWeight - weights_zp[k]) * weights_scale[k];

                            sum_weight += real_weight;
                        }

                outputMaxC = maxIn[0] * sum_weight;
                outputMinC = minIn[0] * sum_weight;
//                if (outputMinC > outputMaxC)
//                //could happen if weight is negative
//                {
//                    auto temp = outputMaxC;
//                    outputMaxC = outputMinC;
//                    outputMinC = temp;
//                }
                if (hasBias)
                {
                    real_bias = ((int64_t) bias->at(k)) * biasScale;
                    outputMinC += real_bias;
                    outputMaxC += real_bias;
                }

                if (opIt->hasAttr("leakyAlpha"))
                {
                    auto alpha = opIt->get<double>("leakyAlpha");
                    if (outputMinC < 0)
                        outputMinC = outputMinC*alpha;
                }

                outMax[k] = outputMaxC;
                outMin[k] = outputMinC;
            }
            outputMin = *std::min_element(outMin.begin(), outMin.end());
            outputMax = *std::max_element(outMax.begin(), outMax.end());

            calcZeroPointAndScalePerTensor(outputMax, outputMin, outScale[0], outZp[0]);

            mv::QuantizationParams newOutputQuantization = {outZp,outScale,{outputMin},{outputMax}};
            output->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
            opIt->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
        }
    }
}

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(DecideOutputDataType)
        .setFunc(decideOutputDataType)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );
    }
}

void decideOutputDataType(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    bool PredictionOfQuantizationOutput;
    auto returnedParams = model.getGlobalConfigParams();
    if (returnedParams->hasAttr("PredictionOfQuantizationOutput"))
        PredictionOfQuantizationOutput = returnedParams->get<bool>("PredictionOfQuantizationOutput");
    bool inputQuantized, weightsQuantized;

   if (PredictionOfQuantizationOutput)
        updateOutputQuantParams(pass, model);
    else
    {
        auto convs = om.getOps("Conv");
        bool outputConvHasQuantParams, outputConvHasEmptyQuantParams, outputConvHasNeutralQuantParams;
        for (auto conv : convs)
        {
            inputQuantized = false, weightsQuantized = false;
            outputConvHasQuantParams = conv->getOutputTensor()[0]->hasAttr("quantParams");
            if (outputConvHasQuantParams)
            {
                outputConvHasEmptyQuantParams =
                    conv->getOutputTensor()[0]->get<mv::QuantizationParams>("quantParams").isEmpty();
                outputConvHasNeutralQuantParams =
                    conv->getOutputTensor()[0]->get<mv::QuantizationParams>("quantParams").isNeutral();
            }
            if (!outputConvHasQuantParams|| outputConvHasEmptyQuantParams || outputConvHasNeutralQuantParams)
            {
                if (conv->getInputTensor()[0]->hasAttr("quantParams"))
                {
                    if (!(conv->getInputTensor()[0]->get<mv::QuantizationParams>("quantParams").isNeutral() ||
                            conv->getInputTensor()[0]->get<mv::QuantizationParams>("quantParams").isEmpty()))
                    {
                        inputQuantized = true;
                    }
                }
                if (conv->getInputTensor()[1]->hasAttr("quantParams"))
                {
                    if (!(conv->getInputTensor()[1]->get<mv::QuantizationParams>("quantParams").isNeutral() ||
                            conv->getInputTensor()[1]->get<mv::QuantizationParams>("quantParams").isEmpty()))
                    {
                        weightsQuantized = true;
                    }
                }
            }
            if (weightsQuantized && inputQuantized)
            {
                conv->getOutputTensor()[0]->set<mv::DType>("dType", mv::DType("Int32"));
                conv->set<mv::DType>("dType", mv::DType("Int32"));
            }
        }
    }

}
