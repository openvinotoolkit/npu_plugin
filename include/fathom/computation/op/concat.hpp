#ifndef CONCAT_HPP_
#define CONCAT_HPP_

#include "include/fathom/computation/op/multisink_op.hpp"
#include "include/fathom/computation/op/source_op.hpp"


namespace mv
{
    /// \todo Add assertions (dimensions)   
    class Concat : public MultiSinkOp, public SourceOp
    {

    public:

        Concat(const string &name) :
        ComputationOp(OpType::Concat, name),
        MultiSinkOp(OpType::Concat, 2, name),
        SourceOp(OpType::Concat, name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

        Tensor getOutputDef()
        {

            if (!validOutputDef_())
                return Tensor();

            auto input0Shape = getInput(0)->getShape();
            if (input0Shape.ndims() != 3)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                    "' because of incorrect shape " + input0Shape.toString() + " of input 0");
                return Tensor();
            }

            dim_type lastDim = input0Shape[2];

            for (byte_type i = 1; i < inputSlots(); ++i)
            {
                auto inputShape = getInput(i)->getShape();
                if (inputShape.ndims() != 3)
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of input " + Printable::toString(i));
                    return Tensor();
                }
                if (inputShape[0] != input0Shape[0] || inputShape[1] != input0Shape[1])
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of inconsistent inputs shapes " + input0Shape.toString() + " and " + inputShape.toString());
                    return Tensor();
                }

                lastDim += inputShape[2];

            }

            return Tensor(getOutputName(), Shape(input0Shape[0], input0Shape[1], lastDim), getInput(0)->getDType(), getInput(0)->getOrder());
            
        }

    };

}

#endif // CONCAT_HPP_