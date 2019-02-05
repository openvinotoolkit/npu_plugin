#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include <math.h>

static void quantizationFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(Quantization)
        .setFunc(quantizationFnc)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "Set quantizaed weight tables for HW convolutions"
        );
    }
}

template <class T, class R>
std::vector<R> extendToK(size_t size, std::vector<T> value)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<R>(size, (R) value[0] , 0);

    if (value.size() == size)
        return std::vector<R>(value.begin(), value.end());

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimentions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

template <class T>
std::vector<double> convertToDoubleVector(std::vector<T> input)
{
    std::vector<double> result(input.begin(), input.end());
    return result;
}

// Quantization for Gemmlowp output
// S1 = weight scale
// S2 = input activation scale
// S3 = output activation scale
// m  = (S1 * S2)/S3, scale for MAC output
// zeroPointScaled = output zero point scaled to MAC output precision
// biasScaled = bias scaled to MAC output precision
void quantizationFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    std::cout << "HW Quantization Optimization Started" << std::endl;
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (opIterator->getOpType() == "Conv")
        {
            //TODO Need to check if it's running on HW
            //  In POC compiler, Conv is by default set to Hardwarizeable for KMB
            //TODO currently poc compiler assumes 1 output/input for convolution, could it be otherwise?

            auto output = opIterator->getOutputTensor(0);
            auto input = opIterator->getInputTensor(0);

            if (!output->hasAttr("quantizationParams") || !input->hasAttr("quantizationParams") ||
                !output->isQuantized() || !input->isQuantized())
                continue;

            auto outputChannels = output->getShape()[2];

            auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
            std::vector<float> S2 = extendToK<double, float>(outputChannels, inputQuantization.getScale());

            auto outputQuantization = output->get<mv::QuantizationParams>("quantizationParams");
            std::vector<float> S3 = extendToK<double, float>(outputChannels, outputQuantization.getScale());

            std::vector<int32_t> zeroPoint = extendToK<unsigned, int32_t>(outputChannels, outputQuantization.getZeroPoint());

            auto m = S2;
            if (opIterator->inputSlots() > 1)
            {
                auto weights = opIterator->getInputTensor(1);
                auto weightsQuantization = weights->get<mv::QuantizationParams>("quantizationParams");
                std::vector<float> S1 = extendToK<double, float>(outputChannels,weightsQuantization.getScale());
                //S1*S2
                std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
            }

            //TODO: Fuse ReLU into quantization (i.e. make ReLU == saturation), shouldn't this be done as a different pass?
            // m / S3
            std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());

            //TODO need to handle 16bits case - per alessandro bias need to be converted to int32
            auto bits = output->getDType().getSizeInBytes() * 8;
            auto shift = 2*bits - 1;
            auto intScale = pow(2, shift);

            std::vector<int16_t> mScaled(m.size());

            //m*intScaled
            for (unsigned i = 0; i < m.size(); ++i)
                mScaled[i] = (int16_t) (m[i] * intScale);

            std::vector<int32_t> zeroPointScaled(m.size());
            std::transform(zeroPoint.begin(), zeroPoint.end() , m.begin(), zeroPointScaled.begin(), std::divides<float>());

            if (opIterator->hasAttr("bias"))
            {
                auto bias = dm.getTensor(opIterator->get<std::string>("bias"));
                auto data = bias->getData();
                //auto biasQuantization = bias->get<mv::QuantizationParams>("quantizationParams");
                //auto Z_bias = biasQuantization.getZeroPoint();
                //auto S_bias = biasQuantization.getScale();
                std::transform(data.begin(), data.end(), zeroPointScaled.begin(), data.begin(), std::plus<int32_t>());
                bias->populate(data);
                bias->setDType(mv::DType("Int32"));
            }
            else
            {
                mv::Order order(mv::Order::getColMajorID(1));
                const std::string biasTensorName = opIterator->getName() + "_bias";
                mv::Shape shape({outputChannels});

                auto biasTensor = dm.defineTensor(biasTensorName, shape, mv::DType("Int32"), order, convertToDoubleVector<int32_t>(zeroPointScaled));
                om.addAttr(opIterator, "bias", biasTensor->getName());
            }

            mv::Shape shape({outputChannels, 1, 1, 4});

            auto bias = dm.getTensor(opIterator->get<std::string>("bias"));
            auto biasData = bias->getData();

            std::vector<int32_t> weightsTableData(shape.totalSize());
            // per channel layout:
            // 3 -> bias
            // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
            // 1 -> SP_PTR
            // 0 -> DATA_PTR
            // TODO mult & prelu are currently not implemented
            for (size_t i = 0; i < weightsTableData.size(); i+=4)
            {
                weightsTableData[i+2] = ((int32_t)mScaled[i/4] << 16) | shift << 2;
                weightsTableData[i+3] = biasData[i/4];
            }

            auto weightTableTensor = dm.defineTensor(opIterator->getName() + "_weights_table", shape, mv::DType("Int32"), mv::Order("NWHC"),
                convertToDoubleVector<int32_t>(weightsTableData));
            om.addAttr(opIterator, "weights_table", weightTableTensor->getName());
        }

    }

    std::cout << "HW Quantization Optimization Ended" << std::endl;
}
