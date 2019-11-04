#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value);
int computeAppropriatePadding(mv::Data::TensorIterator tensor);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeTensorsQuantParams)
        .setFunc(computeTensorsQuantParams)
        .setDescription(
            "This pass computes the appropriate quantize params extends and prepares them for serialization."
        );
    }
}

int computeAppropriatePadding(mv::Data::TensorIterator tensor)
{
    int pad;
    if (tensor->getDType() == mv::DType("Float16"))
        pad = 8;
    else if (tensor->getDType() == mv::DType("UInt8"))
        pad = 16;
    return pad;
}

void computeTensorsQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
         std::string opType = opIt->getOpType();
         std::string taskOp;
         if (opIt->getOpType() ==  "DPUTask")
         {
             taskOp = opIt->get<std::string>("taskOp");
             bool isElementWise = (taskOp == "Eltwise");
             bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
             if (isConv || taskOp == "MaxPool" ||  isElementWise)
             {
                auto output = opIt->getOutputTensor(0);
                auto input = opIt->getInputTensor(0);
                auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
                int pad = computeAppropriatePadding(opIt->getOutputTensor(0));
                outputChannels = mv::round_up(outputChannels, pad);

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

                     auto scale = extendToK(outputChannels, inputQuantization.getScale());
                     std::vector<float> S2(scale.begin(), scale.end());

                     mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                     scale = extendToK(outputChannels, outputQuantization.getScale());
                     std::vector<float> S3(scale.begin(), scale.end());

                     auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint());
                     std::vector<int32_t> zeroPoint(zeroPointU.begin(), zeroPointU.end());

                     bool isPooling = taskOp == "MaxPool";
                     //Workaround for HW bug #227
                     if (isPooling)
                     {
                         auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint());
                         std::vector<int32_t> inputZeroPoint(inZP.begin(), inZP.end());
                         std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
                     }

                     auto m = S2;

                     // TODO: Fix for multiply
                     if (opIt->hasAttr("hasWeights") && opIt->get<bool>("hasWeights"))
                     {
                         auto weights = opIt->getInputTensor(1);
                         auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                         scale = extendToK(outputChannels, weightsQuantization.getScale());
                         std::vector<float> S1(scale.begin(), scale.end());
                         //S1*S2
                         std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
                     }
                     else if (isElementWise) //Add Subtract
                     {
                         auto input2 = opIt->getInputTensor(1);
                         auto& input2Quantization = input2->get<mv::QuantizationParams>("quantParams");
                         auto input1Scale = inputQuantization.getScale();
                         auto input2Scale = input2Quantization.getScale();
                         if (input1Scale != input2Scale)
                            throw mv::RuntimeError(om, opIt->getName() + ": different values of scales for Add/Subtract is not supported!"
                                                   + std::to_string(input1Scale[0]) + " " + std::to_string(input2Scale[0]));
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
}

template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}
