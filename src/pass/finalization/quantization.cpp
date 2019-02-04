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

mv::Tensor extendToK(std::string name, mv::Shape shape, mv::DType dtype, double value)
{
    std::vector<double> data = mv::utils::generateSequence<double>(shape.totalSize(), value , 0);
    mv::Order order(mv::Order::getRowMajorID(shape.ndims()));
    mv::Tensor t(name, shape, dtype, order, data);
    return t;
}

mv::Tensor extendToK(std::string name, mv::Shape shape, mv::DType dtype, std::vector<double> value)
{
    if (value.size() == 1)
        return extendToK(name, shape, dtype, value[0]);

    size_t outputChannels = shape[0];
    if (value.size() == outputChannels)
    {
        mv::Order order(mv::Order::getRowMajorID(shape.ndims()));
        mv::Tensor t(name, shape, dtype, order, value);
        return t;
    }

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimentions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}
template <class T>
std::vector<T> extendToK(size_t size, std::vector<T> value)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<T>(size, value[0] , 0);

    if (value.size() == size)
        return value;

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
            //In POC compiler, Conv is by default set to Hardwarizeable for KMB
            //TODO currently poc compiler assumes 1 output/input for convolution, could it be otherwise?

            auto output = opIterator->getOutputTensor(0);
            auto input = opIterator->getInputTensor(0);

            if (!output->hasAttr("quantizationParams") || !input->hasAttr("quantizationParams") ||
                !output->isQuantized() || !input->isQuantized())
                continue;

            auto outputChannels = output->getShape()[2];


            auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
            std::vector<float> S2 = extendToK<float>(outputChannels, inputQuantization.getScale());

            auto outputQuantization = output->get<mv::QuantizationParams>("quantizationParams");
            std::vector<float> S3 = extendToK<float>(outputChannels, outputQuantization.getScale());

            std::vector<int64_t> zeroPoint = extendToK<int64_t>(outputChannels, outputQuantization.getZeroPoint());

            auto m = S2;

            if (opIterator->inputSlots() > 1)
            {
                auto weights = opIterator->getInputTensor(1);
                auto weightsQuantization = weights->get<mv::QuantizationParams>("quantizationParams");
                std::vector<float> S1 = extendToK<float>(outputChannels,weightsQuantization.getScale());
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
                mScaled[i] = (int16_t) m[i] * intScale;

            std::vector<int32_t> zeroPointScaled(m.size());
            std::transform(zeroPoint.begin(), zeroPoint.end() , m.begin(), zeroPointScaled.begin(), std::divides<int32_t>());

            //auto shiftExt = extendToK("shift", shape, mv::DType("UInt8"), shift);

            if (opIterator->hasAttr("bias"))
            {
                auto bias = dm.getTensor(opIterator->get<std::string>("bias"));
                auto data = bias->getData();
                std::transform(data.begin(), data.end(), zeroPointScaled.begin(), data.begin(), std::plus<int32_t>());
                bias->populate(data);
                bias->setDType(mv::DType("Int32")); //TODO is this  ok?
            }
            else
            {
                //TODO verify order and shape
                mv::Order order(mv::Order("W"));
                const std::string name = opIterator->getName() + "_bias";
                mv::Shape shape({outputChannels});
                mv::Tensor t(name, shape, mv::DType("Int32"), order, convertToDoubleVector<int32_t>(zeroPointScaled));

                opIterator->set<mv::Tensor>("bias", t);
            }

            // per channel layout:
            // 3 -> bias
            // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
            // 1 -> SP_PTR
            // 0 -> DATA_PTR
            // TODO mult & prelu are currently not implemented
            mv::Shape shape({outputChannels, 1, 1, 4});

            auto bias = dm.getTensor(opIterator->get<std::string>("bias"));
            auto bias_data = bias->getData();

            std::vector<int32_t> weights_table_data(shape.totalSize());
            for (size_t i = 0; i < weights_table_data.size(); i+=4)
            {
                weights_table_data[i+2] = (mScaled[i/4] << 16) | shift << 2;
                weights_table_data[i+3] = bias_data[i/4];
            }
            //TODO check order
            mv::Tensor t(opIterator->getName() + "_weights_table", shape, mv::DType("Int32"), mv::Order(mv::Order::getColMajorID(4)),
                convertToDoubleVector<int32_t>(weights_table_data));
            opIterator->set<mv::Tensor>("weights_table", t);
        }

    }

    std::cout << "HW Quantization Optimization Ended" << std::endl;
}
