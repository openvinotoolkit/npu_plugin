#ifndef MAXPOOL_HPP_
#define MAXPOOL_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"
#include "include/fathom/computation/op/sink_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)
    class MaxPool2D : public KernelOp, public SinkOp
    {

    public:

        MaxPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
        ComputationOp(OpType::MaxPool2D, name),
        KernelOp(OpType::MaxPool2D, stride, padding, name),
        SinkOp(OpType::MaxPool2D, 1, name)
        {
            addAttr("kSize", AttrType::UnsignedVec2DType, kernelSize);
            addAttr("executable", AttrType::BoolType, true);
        }

        Tensor getOutputDef(byte_type idx)
        {

            if (idx > 0)
                return Tensor();

            if (!validOutputDef_())
                return Tensor();

            auto input = getInput(0);
            auto inputShape = input->getShape();

            if (inputShape.ndims() != 3)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of input");
                return Tensor();
            }

            auto padding = getAttr("padding").getContent<UnsignedVector4D>();
            auto stride = getAttr("stride").getContent<UnsignedVector2D>();
            auto kSize = getAttr("kSize").getContent<UnsignedVector2D>();
            
            if (inputShape[0] + padding.e0 + padding.e1 < kSize.e0)
            {
                logger_.log(Logger::MessageType::MessageError, 
                    "Unable to define output tensor for '" + name_ + 
                    "' because of pooling kernel width (" + Printable::toString(kSize.e0) + 
                    ") larger than padded input width (" + Printable::toString(inputShape[0] + padding.e0 + padding.e1) + ")");

                return Tensor();
            }

            if (inputShape[1] + padding.e2 + padding.e3 < kSize.e1)
            {
                logger_.log(Logger::MessageType::MessageError, 
                    "Unable to define output tensor for '" + name_ + 
                    "' because of pooling kernel height (" + Printable::toString(kSize.e1) + 
                    ") larger than padded input height (" + Printable::toString(inputShape[1] + padding.e2 + padding.e3) + ")");

                return Tensor();
            }

            Shape outputShape((inputShape[0] + padding.e0 + padding.e1 - kSize.e0) / stride.e0 + 1, (
                inputShape[1] + padding.e2 + padding.e3 - kSize.e1) / stride.e1 + 1, inputShape[2]);

            return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

        }

    };

}

#endif // MAXPOOL_HPP_