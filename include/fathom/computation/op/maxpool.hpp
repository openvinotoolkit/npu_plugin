#ifndef MAXPOOL_HPP_
#define MAXPOOL_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"
#include "include/fathom/computation/op/sink_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)
    class MaxPool : public KernelOp, public SinkOp
    {

    public:

        MaxPool(const Logger &logger, Shape kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
        ComputationOp(logger, "maxpool", name),
        KernelOp(logger, "maxpool", strideX, strideY, padX, padY, name),
        SinkOp(logger, "maxpool", name)
        {
            addAttr("kSize", AttrType::ShapeType, kernelShape);
        }

        UnpopulatedTensor getOutputDef()
        {

            if (!validOutputDef_())
                return UnpopulatedTensor(logger_);

            auto input = getInput(0);
            auto inputShape = input->getShape();

            if (inputShape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of input");
                return UnpopulatedTensor(logger_);
            }

            auto padX = getAttr("padX").getContent<byte_type>();
            auto padY = getAttr("padY").getContent<byte_type>();
            auto strideX = getAttr("strideX").getContent<byte_type>();
            auto strideY = getAttr("strideY").getContent<byte_type>();

            auto kShape = getAttr("kSize").getContent<Shape>();

            if (kShape.ndims() != 2)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect kernel shape " + kShape.toString());
                return UnpopulatedTensor(logger_);
            }

            Shape outputShape(inputShape[0], getOutputDim_(inputShape[1], kShape[0], padX, strideX), getOutputDim_(inputShape[2], kShape[1], padY, strideY), inputShape[3]);

            return UnpopulatedTensor(logger_, getOutputName(), outputShape, input->getDType(), input->getOrder());

        }

    };

}

#endif // MAXPOOL_HPP_