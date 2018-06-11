#ifndef CONV_HPP_
#define CONV_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"
#include "include/fathom/computation/op/multisink_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)   
    class Conv : public KernelOp, public MultiSinkOp
    {

    public:

        Conv(UnsignedVector2D stride, UnsignedVector4D padding, const string& name) :
        ComputationOp(OpType::Conv2D, name),
        KernelOp(OpType::Conv2D, stride, padding, name),
        MultiSinkOp(OpType::Conv2D, 2, name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

        Tensor getOutputDef()
        {

            if (!validOutputDef_())
                return Tensor();

            auto input = getInput(0);
            auto inputShape = input->getShape();
            auto weights = getInput(1);
            auto weightsShape = weights->getShape();

            if (inputShape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                    "' because of incorrect shape " + inputShape.toString() + " of input");
                return Tensor();
            }
            
            if (weightsShape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of weights");
                return Tensor();
            }

            if (inputShape[3] != weightsShape[2])
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                    "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[3])
                    + ") and weights (" + Printable::toString(weightsShape[2]) + ")");
                return Tensor();
            }

            auto padding = getAttr("padding").getContent<UnsignedVector4D>();
            auto stride = getAttr("stride").getContent<UnsignedVector2D>();

            if (inputShape[1] + padding.e0 + padding.e1 < weightsShape[0])
            {
                logger_.log(Logger::MessageType::MessageError, 
                    "Unable to define output tensor for '" + name_ + 
                    "' because of filter kernel width (" + Printable::toString(weightsShape[0]) + 
                    ") larger than padded input width (" + Printable::toString(inputShape[1] + padding.e0 + padding.e1) + ")");

                return Tensor();
            }

            if (inputShape[2] + padding.e2 + padding.e3 < weightsShape[1])
            {
                logger_.log(Logger::MessageType::MessageError, 
                    "Unable to define output tensor for '" + name_ + 
                    "' because of filter kernel height (" + Printable::toString(weightsShape[1]) + 
                    ") larger than padded input height (" + Printable::toString(inputShape[2] + padding.e2 + padding.e3) + ")");

                return Tensor();
            }
            
            // Make sure that the result of subtract will not be negative
            Shape outputShape(inputShape[0], (inputShape[1] + padding.e0 + padding.e1 - weightsShape[0]) / stride.e0 + 1, (
                inputShape[2] + padding.e2 + padding.e3 - weightsShape[1]) / stride.e1 + 1, weightsShape[3]);

            return Tensor(getOutputName(), outputShape, input->getDType(), input->getOrder());

        }

    };

}

#endif // CONV_HPP_