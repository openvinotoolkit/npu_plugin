#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateOutputQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateOutputQuantParamsTasos(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeTensorsQuantParams)
        .setFunc(computeTensorsQuantParams)
        .setDescription(
            "This pass computes the appropriate quantize params extends and prepares them for serialization."
        );

        MV_REGISTER_PASS(UpdateOutputQuantParams)
        .setFunc(updateOutputQuantParamsTasos)
        .setDescription(
            "The pass will estimate output tensor quantization param where quantization is needed."
        );
    }
}

bool isQuantizationParamNeutral(mv::QuantizationParams& quants)
{
    auto scale = quants.getScale();
    for (size_t i = 0; i < scale.size(); i++)
    {
        if (scale[i] != 1.0)
            return false;
    }
    auto zp = quants.getZeroPoint();
    for (size_t i = 0; i < zp.size(); i++)
    {
        if (zp[i] != 0)
            return false;
    }
    return true;
}

void updateOutputQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");

    for(auto& opIt : dpuTasks)
    {
         std::string taskOp = opIt->get<std::string>("taskOp");
         bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
         if (isConv || taskOp == "MaxPool")
         {
            auto output = opIt->getOutputTensor(0);
            auto input = opIt->getInputTensor(0);
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = mv::round_up(outputChannels, 16);

            std::vector<int> shift(outputChannels, 0);
            std::vector<int16_t> mScaled(outputChannels, 0);

            if (!output->hasAttr("quantParams") || isQuantizationParamNeutral(output->get<mv::QuantizationParams>("quantParams")))
            {

                auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                double inf = std::numeric_limits<double>::infinity();

                auto weights = opIt->getInputTensor(1);
                auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                auto weights_scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
                auto weights_zp = extendToK(outputChannels, weightsQuantization.getZeroPoint(), weights->getName());
                std::vector<double> outMin(outputChannels, inf);
                std::vector<double> outMax(outputChannels, -inf);
                std::vector<double> outScale(outputChannels);
                std::vector<int64_t> outZp(outputChannels);

                auto kernelSize = weights->getShape().totalSize()/outputChannels;

                auto minIn = extendToK(outputChannels, inputQuantization.getMin(), input->getName());
                auto maxIn = extendToK(outputChannels, inputQuantization.getMax(), input->getName());
                bool hasBias = opIt->hasAttr("bias");
                mv::Data::TensorIterator bias;
                if (hasBias)
                {
                    bias = dm.getTensor(opIt->get<std::string>("bias"));
                }
                size_t index = 0;
                for (size_t c = 0; c < outputChannels; c++)
                {
                    for (size_t i = 0; i < kernelSize; i++)
                    {
                        auto real_weight = ((int64_t) weights->at(index) - weights_zp[c]) * weights_scale[c];
                        auto localMin = minIn[c] * real_weight;
                        auto localMax = maxIn[c] * real_weight;
                        if (hasBias)
                        {
                            localMin += (int64_t) bias->at(c);
                            localMax += (int64_t) bias->at(c);
                        }
                        if (localMin < outMin[c])
                            outMin[c] = localMin;
                        if (localMax > outMax[c])
                            outMax[c] = localMax;
                        index++;
                    }
                    outScale[c] = (outMax[c] - outMin[c])/255;
                    std::cout << " channel " << std::to_string(c) << " scale " << std::to_string(outScale[c]) << std::endl;
                    outZp[c] = floor(outMin[c]/255);
                    std::cout << " channel " << std::to_string(c) << " zp " << std::to_string(outZp[c]) << std::endl;

                }
            }
        }

    }
}

void updateOutputQuantParamsTasos(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
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

    auto dpuTasks = om.getOps("DPUTask");

    for(auto& opIt : dpuTasks)
    {
        //Pool will just propagate...
         std::string taskOp = opIt->get<std::string>("taskOp");
         bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
         if (isConv)
         {
            auto output = opIt->getOutputTensor(0);
            auto input = opIt->getInputTensor(0);
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];

            std::vector<int> shift(outputChannels, 0);
            std::vector<int16_t> mScaled(outputChannels, 0);
            double inf = std::numeric_limits<double>::infinity();
            double outputMin = inf;
            double outputMax = -inf;

            std::vector<double> outMin(outputChannels, inf);
            std::vector<double> outMax(outputChannels, -inf);
            if (!output->hasAttr("quantParams")
                    || isQuantizationParamNeutral(output->get<mv::QuantizationParams>("quantParams")))
            {

                auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                double inf = std::numeric_limits<double>::infinity();

                auto weights = opIt->getInputTensor(1);
                auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                auto weights_scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
                auto weights_zp = extendToK(outputChannels, weightsQuantization.getZeroPoint(), weights->getName());

                //input/output quantization are per tensor, weights quantization are per channel
                std::vector<double> outScale(1);
                std::vector<int64_t> outZp(1);

                auto fullkernelSize = weights->getShape().totalSize()/outputChannels;

                auto minIn = inputQuantization.getMin();
                auto maxIn = inputQuantization.getMax();
                std::cout << " minIN " << std::to_string(minIn[0]) << std::endl;
                std::cout << " maxIN " << std::to_string(maxIn[0]) << std::endl;
                //auto minWeights = extendToK(outputChannels, weightsQuantization.getMin(), weights->getName());
                //auto maxWeights = extendToK(outputChannels, weightsQuantization.getMax(), weights->getName());

                bool hasBias = opIt->hasAttr("bias");
                mv::QuantizationParams biasQuantization({},{},{},{});
                mv::Data::TensorIterator bias;
                std::vector<double> biasScale;
                std::vector<uint64_t> biasZp;
                if (hasBias)
                {
                    bias = dm.getTensor(opIt->get<std::string>("bias"));
                    biasQuantization = bias->get<mv::QuantizationParams>("quantParams");
                }
                double_t real_weight, real_bias;
                for (size_t c = 0; c < outputChannels; c++)
                {
                    double sum_weight = 0;
//                    outMax[c] = maxIn[c] * fullkernelSize * maxWeights[c];
//                    outMin[c] = minIn[c] * fullkernelSize * minWeights[c];
                    double outputMinC = 0;
                    double outputMaxC = 0;
                    for (size_t index = 0; index < fullkernelSize; index++)
                    {
                        real_weight = ((int64_t) weights->at(index + c*fullkernelSize) - weights_zp[c]) * weights_scale[c];
                        if (real_weight < weightsQuantization.getMin()[0])
                            std::cout << " smaller than min : channel " << std::to_string(c) << " real_weight " << std::to_string(index) << " : " << std::to_string(real_weight) << std::endl;
                        if (real_weight > weightsQuantization.getMax()[0])
                            std::cout << " smaller than max : channel " << std::to_string(c) << " real_weight " << std::to_string(index) << " : " << std::to_string(real_weight) << std::endl;
                        sum_weight += real_weight;
                    }
                    outputMaxC = maxIn[0] * sum_weight;
                    outputMinC = minIn[0] * sum_weight;
                    if (outputMinC > outputMaxC)
                    //could happen if weight is negative
                    {
                        auto temp = outputMaxC;
                        outputMaxC = outputMinC;
                        outputMinC = temp;
                    }
                    std::cout << " before bias channel " << std::to_string(c) << " out Min " << std::to_string(outputMinC) << std::endl;
                    std::cout << " before bias channel " << std::to_string(c) << " out Max " << std::to_string(outputMaxC) << std::endl;
                    if (hasBias)
                    {
                        real_bias = ((int64_t) bias->at(c)) * biasQuantization.getScale()[0];
                        std::cout << " channel " << std::to_string(c) << " real_bias " << std::to_string(real_bias) << std::endl;

                        outputMinC += real_bias;
                        outputMaxC += real_bias;
                    }
                    outMax[c] = outputMaxC;
                    outMin[c] = outputMinC;

                    std::cout << " channel " << std::to_string(c) << " out Min " << std::to_string(outputMinC) << std::endl;
                    std::cout << " channel " << std::to_string(c) << " out Max " << std::to_string(outputMaxC) << std::endl;
                    if (outputMinC < outputMin)
                        outputMin = outputMinC;

                    if (outputMaxC > outputMax)
                        outputMax = outputMaxC;
                }

                //avg min/max
                double avgMin = 0;
                double avgMax = 0;
                for (size_t c = 0; c<outputChannels; c++)
                {
                    avgMin += outMin[c];
                    avgMax += outMax[c];
                }
                outputMin = avgMin / outputChannels;
                outputMax = avgMax / outputChannels;

                std::cout <<  " out Min " << std::to_string(outputMin) << std::endl;
                std::cout << " out Max " << std::to_string(outputMax) << std::endl;

                outScale[0] = (outputMax - outputMin)/255;
                std::cout << " scale " << std::to_string(outScale[0]) << std::endl;
                if ((outputMax > 0.0 && outputMin > 0.0)||(outputMax < 0.0 && outputMin < 0.0))
                    outZp[0] = 0;
                else if ((outputMax > 0.0 && outputMin < 0.0))
                    outZp[0] = 128;
                std::cout <<  " zp " << std::to_string(outZp[0]) << std::endl;
            }
        }

    }
}

void computeTensorsQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");

    for(auto& opIt : dpuTasks)
    {
         std::string taskOp = opIt->get<std::string>("taskOp");
         bool isEltwise = taskOp == "Eltwise";
         bool isEltwiseMult = false;
         bool isEltwiseAddSub = false;
         if(isEltwise)
         {
             auto eltwiseType = opIt->get<std::string>("eltwiseType");
             if(eltwiseType == "Add" || eltwiseType == "Subtract")
                 isEltwiseAddSub = true;
             if(eltwiseType == "Multiply")
                 isEltwiseMult = true;
         }
         bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
         if (isConv || taskOp == "MaxPool" || isEltwiseMult || isEltwiseAddSub)
         {
            auto output = opIt->getOutputTensor(0);
            auto input = opIt->getInputTensor(0);
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = mv::round_up(outputChannels, 16);

            std::vector<int> shift(outputChannels, 0);
            std::vector<int16_t> mScaled(outputChannels, 0);

            if (output->hasAttr("quantParams") && input->hasAttr("quantParams") &&
             output->isQuantized() && input->isQuantized())
            {
                 // Quantization for Gemmlowp output
                 // S1 = weight scale
                 // S2 = input activation scale
                 // S3 = output activation scale
                 // m  = (S1 * S2)/S3, scale for MAC output
                 // zeroPointScaled = output zero point scaled to MAC output precision
                 // biasScaled = bias scaled to MAC output precision

                 auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                 //inputQuantization.extendParamsToOutputChannelSize(outputChannels);

                 auto scale = extendToK(outputChannels, inputQuantization.getScale(), input->getName());
                 std::vector<float> S2(scale.begin(), scale.end());

                 mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                 scale = extendToK(outputChannels, outputQuantization.getScale(), output->getName());
                 std::vector<float> S3(scale.begin(), scale.end());

                 auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint(), output->getName());
                 std::vector<int32_t> zeroPoint(zeroPointU.begin(), zeroPointU.end());

                 bool isPooling = taskOp == "MaxPool";
                 //Workaround for HW bug #227
                 if (isPooling)
                 {
                     auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint(), input->getName());
                     std::vector<int32_t> inputZeroPoint(inZP.begin(), inZP.end());
                     std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
                 }

                 auto m = S2;

                 if ((opIt->hasAttr("hasWeights") && opIt->get<bool>("hasWeights")) || isEltwiseMult)
                 {
                     auto weights = opIt->getInputTensor(1);
                     auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                     scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
                     std::vector<float> S1(scale.begin(), scale.end());
                     //S1*S2
                     std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
                 }
                 else if (isEltwiseAddSub) //Add Subtract
                 {
                     auto input2 = opIt->getInputTensor(1);
                     auto& input2Quantization = input2->get<mv::QuantizationParams>("quantParams");
                     auto input1Scale = inputQuantization.getScale();
                     auto input2Scale = input2Quantization.getScale();
                     if (input1Scale != input2Scale)
                        throw mv::RuntimeError(om, opIt->getName() + ": different values of scales for Add/Subtract is not supported!"
                                               + std::to_string(input1Scale[0]) + " " + std::to_string(input2Scale[0]));
                 }

                 //Note: There are PPE Types SIGMOID, TAN, EXP, SQRT, RSQRT, FLEXARB that need their output
                 //quantized to 13-bits, then runtime uses a LUT to correspond to 8-bit
                 if (opIt->hasAttr("postOpTypes"))
                 {
                     auto ppeIterator = std::find(opIt->get<std::vector<std::string>>("postOpTypes").begin(),
                                               opIt->get<std::vector<std::string>>("postOpTypes").end(),
                                               "Sigmoid");
                     if (ppeIterator != opIt->get<std::vector<std::string>>("postOpTypes").end())
                     {
                        auto ppeQuantumBits = 5;
                        auto ppeQuantum = std::pow(2, ppeQuantumBits);
                        std::transform(m.begin(), m.end(), m.begin(), std::bind(std::multiplies<float>(),
                                                                                std::placeholders::_1, ppeQuantum));
                     }
                }
                 // Fuse ReLU into quantization (i.e. make ReLU == saturation), will be done using a separate pass

                 // m / S3
                 std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());

                 //TODO need to handle 16bits case - per Alessandro bias need to be converted to int32
                 auto bits = 15;
                 auto mSize = m.size();
                 int exponent;
                 double mantissa;

                 for (size_t i = 0; i < mSize; i++)
                 {
                     mantissa = std::frexp(m[i], &exponent);
                     shift[i] = bits - exponent;
                     mScaled[i] = (mantissa * pow(2, bits));
                 }
                 std::vector<int32_t> zeroPointScaled(m.size());
                 std::transform(zeroPoint.begin(), zeroPoint.end() , m.begin(), zeroPointScaled.begin(), std::divides<float>());

                 std::vector <unsigned> ser_shift = std::vector<unsigned>(shift.begin(), shift.end());
                 std::vector <unsigned> ser_scale = std::vector<unsigned>(mScaled.begin(), mScaled.end());
                 outputQuantization.quantize(ser_shift, ser_scale);
            }
        }

    }
}

template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<T>(size, static_cast<T>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}
