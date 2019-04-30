#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <numeric>
#include <cmath>

static void quantizeLayers(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(QuantizeLayers)
        .setFunc(quantizeLayers)
        .setDescription(
            "This pass computes the appropriate quantize params extends and prepares them for serialization."
        );
    }
}

void quantizeLayers(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
         std::string opType = opIt->getOpType();
         if (opType == "Conv" || opType == "DepthwiseConv" || opType == "MaxPool")
         {
             auto output = opIt->getOutputTensor(0);
             size_t outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
             std::vector<int32_t> outputZeroPoint, inputZeroPoint, resultZeroPoint;
             std::vector<double> macScale, division, mantissa_v, mScaled, zeroPointScaled;
             std::vector<int> exponent_v, bits(outputChannels, 15), shift;
             double mantissa;
             int exponent;
             //Output without input impossible
            if (output->hasAttr("quantParams"))
            {
                auto outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                outputQuantization.extendParamsToOutputChannelSize(outputChannels);
                outputZeroPoint = std::vector<int32_t>(outputQuantization.getZeroPoint().begin(), outputQuantization.getZeroPoint().end());
//                output->set<mv::QuantizationParams>("quantParams", outputQuantization);
                auto input = opIt->getInputTensor(0);
                auto inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                inputQuantization.extendParamsPartialToOutputChannelSize(outputChannels);
                if (opType == "MaxPool")
                {
                    inputQuantization.extendParamsToOutputChannelSize(outputChannels);
                    inputZeroPoint = std::vector<int32_t>(inputQuantization.getZeroPoint().begin(), inputQuantization.getZeroPoint().end());
                    std::transform(outputZeroPoint.begin(), outputZeroPoint.end(), inputZeroPoint.begin(), std::back_inserter(outputZeroPoint)
                                   , std::minus<int32_t>());
                }
//                input->set<mv::QuantizationParams>("quantParams", inputQuantization);
                //WEIGHTS
                std::vector<double> weightTensorScale(outputChannels, 1);
                if (opIt->getInputTensor().size() > 1)
                {
                    auto weights = opIt->getInputTensor(1);
                    if (weights->hasAttr("quantParams"))
                    {
                        auto weightQuantization = weights->get<mv::QuantizationParams>("quantParams");
                        weightQuantization.extendParamsPartialToOutputChannelSize(outputChannels);
                        //S1
                        weightTensorScale = weightQuantization.getScale();
//                        weights->set<mv::QuantizationParams>("quantParams", weightQuantization);
                    }
                }
                std::transform(weightTensorScale.begin(), weightTensorScale.end(), inputQuantization.getScale().begin(), std::back_inserter(macScale),
                                                                  std::multiplies<double>());
                std::transform(macScale.begin(), macScale.end(), outputQuantization.getScale().begin(), std::back_inserter(division), std::divides<double>());

                for (auto it = division.begin(); it != division.end(); ++it)
                {
                    mantissa = std::frexp(*it, &exponent);
                    mantissa_v.push_back(mantissa);
                    exponent_v.push_back(exponent);
                }

                std::transform(bits.begin(), bits.end(), exponent_v.begin(), std::back_inserter(shift), std::minus<int>());
                std::vector<double> power_v(outputChannels, pow(2.0, bits[0]));
                std::transform(mantissa_v.begin(), mantissa_v.end(), power_v.begin(), std::back_inserter(mScaled),
                                                                  std::multiplies<double>());
                std::vector<uint16_t> mScaled_conv = std::vector<uint16_t>(mScaled.begin(), mScaled.end());
                std::transform(outputZeroPoint.begin(), outputZeroPoint.end(), division.begin(), std::back_inserter(zeroPointScaled), std::divides<double>());
                std::vector<int32_t> zeroPointScaled_conv = std::vector<int32_t>(zeroPointScaled.begin(), zeroPointScaled.end());
                std::vector <uint8_t> ser_shift = std::vector<uint8_t>(shift.begin(), shift.end());
                std::vector <uint16_t> ser_scale = std::vector<uint16_t>(mScaled_conv.begin(), mScaled_conv.end());
                outputQuantization.quantize(ser_shift, ser_scale);

                if (opIt->hasAttr("bias"))
                {
                    auto biasTensor = dm.getTensor(opIt->get<std::string>("bias"));
                    auto data = biasTensor->getIntData();
                    std::transform(data.begin(), data.end(), zeroPointScaled_conv.begin(), data.begin(), std::plus<int32_t>());
                    biasTensor->setDType(mv::DType("Int32"));
                    biasTensor->populate(data);
                }
                else
                {
                    mv::Order order(mv::Order::getColMajorID(1));
                    const std::string biasTensorName = opIt->getName() + "_bias";
                    mv::Shape shape({outputChannels});
                    std::vector<int64_t> calling_tensor = std::vector<int64_t>(zeroPointScaled_conv.begin(), zeroPointScaled_conv.end());
                    auto biasTensor = dm.defineTensor(biasTensorName, shape, mv::DType("Int32"), order, calling_tensor);
                    om.addAttr(opIt, "bias", biasTensor->getName());
                }
            }
        }
    }
}
