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

        Concat(const Logger &logger, const string &name) :
        ComputationOp(logger, "concat", name),
        MultiSinkOp(logger, "concat", 2, name),
        SourceOp(logger, "concat", name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

        UnpopulatedTensor getOutputDef()
        {

            if (!validOutputDef_())
                return UnpopulatedTensor(logger_);

            auto input0Shape = getInput(0)->getShape();
            if (input0Shape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                    "' because of incorrect shape " + input0Shape.toString() + " of input 0");
                return UnpopulatedTensor(logger_);
            }

            dim_type lastDim = input0Shape[3];

            for (byte_type i = 1; i < inputSlots(); ++i)
            {
                auto inputShape = getInput(i)->getShape();
                if (inputShape.ndims() != 4)
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of input " + Printable::toString(i));
                    return UnpopulatedTensor(logger_);
                }
                if (inputShape[0] != input0Shape[0] || inputShape[1] != input0Shape[1] || inputShape[2] != inputShape[2])
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of inconsistent inputs shapes " + input0Shape.toString() + " and " + inputShape.toString());
                    return UnpopulatedTensor(logger_);
                }

                lastDim += inputShape[3];

            }

            return UnpopulatedTensor(logger_, getOutputName(), Shape(input0Shape[0], input0Shape[1], input0Shape[2], lastDim), getInput(0)->getDType(), getInput(0)->getOrder());
            
        }

    };

}

#endif // CONCAT_HPP_