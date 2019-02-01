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

            mv::Shape shape({outputChannels, 1, 1, 1});
            auto inputQuantization = input->get<mv::QuantizationParams>("quantizationParams");
            auto S2 = extendToK("inputScale", shape, mv::DType("Float32"), convertToDoubleVector<float>(inputQuantization.getScale()));

            auto outputQuantization = output->get<mv::QuantizationParams>("quantizationParams");
            auto S3 = extendToK("outputScale", shape, mv::DType("Float32"), convertToDoubleVector<float>(outputQuantization.getScale()));
            auto zeroPoint = extendToK("outputZeroPoint", shape, mv::DType("Int32"), convertToDoubleVector<int64_t>(outputQuantization.getZeroPoint()));

            auto m = S2;

            if (opIterator->inputSlots() > 1)
            {
                auto weights = opIterator->getInputTensor(1);
                auto weightsQuantization = weights->get<mv::QuantizationParams>("quantizationParams");
                auto S1 = extendToK("weightsScale", shape, mv::DType("Float32"), convertToDoubleVector<float>(weightsQuantization.getScale()));
                m.multiply(S1); // it's elementWise S1*S2 == S2*S1
            }

            //TODO: Fuse ReLU into quantization (i.e. make ReLU == saturation), shouldn't this be done as a different pass?
            m.divide(S3);

            //TODO need to handle 16bits case - per alessandro bias need to be converted to int32
            auto bits = output->getDType().getSizeInBytes() * 8;
            auto shift = 2*bits - 1;
            auto intScale = pow(2, shift);

            auto mScaled = mv::math::multiply(m , intScale);
            mScaled.setDType(mv::DType("Int16")); //TODO need to add functionality to convert values after changing DType

            auto zeroPointScaled = mv::math::divide(zeroPoint , m);
            zeroPointScaled.setDType(mv::DType("Int32")); //TODO need to add functionality to convert values after changing DType

            auto shiftExt = extendToK("shift", shape, mv::DType("UInt8"), shift);

            if (opIterator->hasAttr("bias"))
            {
                auto bias = dm.getTensor(opIterator->get<std::string>("bias"));
                bias->add(zeroPointScaled);
                bias->setDType(mv::DType("Int32"));
            }
            else
            {
                opIterator->set<mv::Tensor>("bias", zeroPointScaled);
            }

            // per channel layout:
            // 3 -> bias
            // 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
            // 1 -> SP_PTR
            // 0 -> DATA_PTR
            // TODO mult & prelu are currently not implemented


        }

    }

    std::cout << "HW Quantization Optimization Ended" << std::endl;
}
