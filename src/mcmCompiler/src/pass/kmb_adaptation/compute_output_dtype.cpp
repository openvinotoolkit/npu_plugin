#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>
#include <functional>

static void decideOutputDataType(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateOutputQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void tensorsToFP16Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void tensorsToU8Fcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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
    std::vector <mv::Data::OpListIterator> convolutions;
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

            calcZeroPointAndScalePerTensor(outputMax, outputMin, 256, mv::getDType(mv::Precision::U8),
                outScale[0], outZp[0]);

            mv::QuantizationParams newOutputQuantization = {outZp,outScale,{outputMin},{outputMax}};
            output->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
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

            bool hasLeakyAlpha = false;
            double leakyAlpha = 1.0; /* 1.0 means no need of multiplication of leakAlpha*/
            bool hasSlopes = false;
            std::vector<double> slopes;

            if (opIt->hasAttr("leakyAlpha"))
            {
                // alpha of LeakyReLU
                hasLeakyAlpha = true;
                leakyAlpha = opIt->get<double>("leakyAlpha");
            }
            else if (opIt->hasAttr("slopes"))
            {
                // slopes of PReLU (channel-wise)
                hasSlopes = true;
            }

            if (hasSlopes)
            {
                slopes = opIt->get<std::vector<double>>("slopes");
                if (slopes.size() == 1)
                {
                    // not expected. It sohuld be converted to leaky relu
                    leakyAlpha = slopes[0];
                    // leakyrely is one case of PReLU: channel shared (leakyRelU) and channel wise
                    // So, consider is as LeakyReLU
                    hasLeakyAlpha = true;
                }
                else if (slopes.size() != kernelShape[mv::KERNEL_OUTPUT_CHANNELS])
                {
                    throw std::runtime_error("The number of slopes does not match with the number of output channels");
                }
            }

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
                if (hasBias)
                {
                    real_bias = ((int64_t) bias->at(k)) * biasScale;
                    outputMinC += real_bias;
                    outputMaxC += real_bias;
                }

                if (hasLeakyAlpha)
                {
                    // leaky relu is applied only negative values
                    if (outputMinC < 0.0)
                    {
                        outputMinC = outputMinC*leakyAlpha;
                    }
                }
                else if (hasSlopes)
                {
                    // This path is to adjust output min range per channel for PReLU
                    // parametric relu is applied only negative values
                    if (outputMinC < 0.0)
                    {
                        double slope = slopes[k];
                        if (slope > 0.0)
                        {
                            outputMinC = outputMinC*slope;
                        }
                        else if (slope < 0.0)
                        {
                            outputMinC *= slope;
                            if (outputMinC > outputMaxC)
                            {
                                outputMaxC = outputMinC;
                            }

                            // no negative values as negative * negative becomes positive value
                            outputMinC = 0.0;
                        }
                    }
                }

                outMax[k] = outputMaxC;
                outMin[k] = outputMinC;
            }
            outputMin = *std::min_element(outMin.begin(), outMin.end());
            outputMax = *std::max_element(outMax.begin(), outMax.end());

            calcZeroPointAndScalePerTensor(outputMax, outputMin, 256, mv::getDType(mv::Precision::U8),
                outScale[0], outZp[0]);

            mv::QuantizationParams newOutputQuantization = {outZp,outScale,{outputMin},{outputMax}};
            output->set<mv::QuantizationParams>("quantParams", newOutputQuantization);
        }
    }
}

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(TensorsToFP16)
        .setFunc(tensorsToFP16Fcn)
        .setDescription(
            "Replaces full precision tensors with FP16 tensors"
        );

        MV_REGISTER_PASS(TensorsToU8)
        .setFunc(tensorsToU8Fcn)
        .setDescription(
            "Replaces quantized int8 tensors with U8 tensors"
        );

        MV_REGISTER_PASS(DecideOutputDataType)
        .setFunc(decideOutputDataType)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );
    }
}

void tensorsToFP16Fcn(const mv::pass::PassEntry&  , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    using namespace mv;
    OpModel om(model);

    auto kernelOp = om.opBegin();
    while (kernelOp != om.opEnd())
    {
        if(kernelOp.outputsSize() > 0)
        {
            auto outputTensor = kernelOp->getOutputTensor(0);
            if(outputTensor->getDType() == mv::DType("Float64") ||
               outputTensor->getDType() == mv::DType("Float32"))
            {
                auto opId = kernelOp->get<unsigned>("opId");
                if (outputTensor->isPopulated())
                {
                    std::vector<double> oldData = kernelOp->getOutputTensor(0)->getDoubleData();
                    std::vector<int64_t> newData(oldData.size());

                    for (size_t i = 0; i < oldData.size(); ++i)
                    {
                        newData[i] = mv::fp32_to_fp16(oldData[i]);
                    }

                    auto kernelShape = kernelOp->getOutputTensor(0)->getShape();
                    auto kernelOrder = kernelOp->getOutputTensor(0)->getOrder();
                    //with data flows I am finding where the op was attached to attache the new one!!!
                    auto outputDataFlows = mv::getOutputDataFlow(om, kernelOp);

                    auto newKernel = om.constantInt("", newData, kernelShape, mv::DType("Float16"), kernelOrder);
                    auto newKernelOp = om.getSourceOp(newKernel);
                    newKernelOp->set<unsigned>("opId", opId);
                    newKernelOp->set<mv::DType>("dType",  mv::DType("Float16"));
                    mv::setOutputDataFlow(om, newKernel, outputDataFlows);
                }
                // In case there is an Input->Conversion sequence then tensor precision doesn't have to be
                // limited to FP16
                else if (kernelOp->getOpType() == "Input" && kernelOp.leftmostOutput().sink()->getOpType() == "Conversion")
                {
                    ++kernelOp;
                }
                else if (td.getTarget() == mv::Target::ma3720)
                {
                    if (outputTensor->getDType() == mv::DType("Float64"))
                        outputTensor->setDType(mv::DType("Float32"));
                    ++kernelOp;
                }
                else
                {
                    outputTensor->setDType(mv::DType("Float16"));
                    ++kernelOp;
                }
            }
            else
                ++kernelOp;
        }
        else
            ++kernelOp;
    }
}

// Pass logic:
// Runtime will handle the input, we uniform all the rest to UInt8
void tensorsToU8Fcn(const mv::pass::PassEntry&  , mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    int64_t zeroPointShift = 128;
    auto sourceDType = mv::DType("Int8");
    auto targetDType = mv::DType("UInt8");

    auto kernelOp = om.opBegin();
    auto inputType = kernelOp->getOutputTensor(0)->getDType();
    auto target = td.getTarget();
    if (inputType == mv::DType("Int8") && target != mv::Target::ma3720) {
       throw std::runtime_error(td.toString(target) + " Compiler doesn't support I8 inputs for the moment, please rescale your data to U8");
    }
    if (inputType == mv::DType("Int8") && target == mv::Target::ma3720) {
        return; //Int8 supported by ma3720 - no need to convert
    }
    for (; kernelOp != om.opEnd(); ++kernelOp)
    {
        if(kernelOp.outputsSize() > 0)
        {
            auto outputTensor = kernelOp->getOutputTensor(0);
            auto outputTensorDType = outputTensor->getDType();
            if(outputTensorDType == sourceDType)
            {
                mv::DType newType = targetDType;
                auto quantParams = outputTensor->get<mv::QuantizationParams>("quantParams");
                auto quantParamsZp = quantParams.getZeroPoint();
                for(auto& zp: quantParamsZp)
                    zp += zeroPointShift;
                quantParams = mv::QuantizationParams(quantParamsZp, quantParams.getScale(),{},{});
                outputTensor->setQuantParams(quantParams);
                outputTensor->setDType(newType);
                if (outputTensor->isPopulated())
                    for(unsigned i = 0; i < outputTensor->size(); ++i)
                        outputTensor->at(i) += zeroPointShift;
            }
        }
    }
}

void decideOutputDataType(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    auto returnedParams = model.getGlobalConfigParams();

    if (returnedParams->hasAttr("PredictionOfQuantizationOutput") &&
        returnedParams->get<bool>("PredictionOfQuantizationOutput")) {
        updateOutputQuantParams(pass, model);
    } else {
        for (auto& p : om.getOpsOfTypes({"Conv", "DepthwiseConv", "MaxPool", "Eltwise"})) {
            for (auto& op : p.second) {
                const auto& opType = p.first;

                // TODO: Can we encapsulate isEmpty() inside isNeutral()?
                bool inputQuantized = true;
                for (size_t i = 0; i < (opType == "Eltwise" ? 2 : 1); ++i) {
                    inputQuantized &= !op->getInputTensor(i)->getQuantParams().isEmpty() &&
                                      !op->getInputTensor(i)->getQuantParams().isNeutral();
                }
                bool outputQuantized = !op->getOutputTensor()[0]->getQuantParams().isNeutral() &&
                                       !op->getOutputTensor()[0]->getQuantParams().isEmpty();

                if (op->getOutputTensor()[0]->getDType() ==  mv::DType("Float16")) {
                    if (returnedParams->hasAttr("FloatOutput") && returnedParams->get<bool>("FloatOutput")) {
                        op->set<bool>("floatPrecision", true);
                    }
                } else if (inputQuantized && !outputQuantized) {
                    if (returnedParams->hasAttr("Int32Output") && returnedParams->get<bool>("Int32Output")) {
                        op->getOutputTensor()[0]->setDType(mv::DType("Int32"));
                    }
                    // NOTE: HW limitation, in mixed mode the grids of the MPEs are conflicting between
                    // each other, which leads to 1x1 workloads, so we will do an explicit conversion
                    // in different cases
                    if (returnedParams->hasAttr("FloatOutput") && returnedParams->get<bool>("FloatOutput")) {
                        if (op->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] == 1 &&
                            op->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 1) {
                            if (td.getTarget() != mv::Target::ma3720)
                                op->set<bool>("mixedToFloat", true);
                            op->getOutputTensor()[0]->setDType(mv::DType("Float16"));
                        } else {
                            const auto& channelScale = op->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale();
                            bool perTensor = std::all_of(channelScale.begin(), channelScale.end(),
                                                         [&](double x) {
                                                             return std::abs(x - channelScale[0]) <= 0.01f;
                                                         });
                            if (perTensor) {
                                op->set<bool>("placeConversionToFloat", true);
                            }
                        }
                    }
                }
            }
        }
    }
}
