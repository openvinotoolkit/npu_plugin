#ifndef BIAS_OP_HPP_
#define BIAS_OP_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace Op
    {

        /// \todo Add assertions (dimensions)   
        class Bias : public SourceOp, public SinkOp
        {

        public:

            Bias(const string& name) :
            ComputationOp(OpType::Bias, name),
            SourceOp(OpType::Bias, 1, name),
            SinkOp(OpType::Bias, 2, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

            Tensor getOutputDef(byte_type idx)
            {

                if (idx > 0)
                    return Tensor();

                if (!validOutputDef_())
                    return Tensor();

                auto input = getInputTensor(0);
                auto inputShape = input->getShape();
                auto biases = getInputTensor(1);
                auto biasesShape = biases->getShape();
                
                if (biasesShape.ndims() != 1)
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                            "' because of incorrect shape " + biasesShape.toString() + " of biases (must be a vector)");
                    return Tensor();
                }

                if (inputShape[-1] != biasesShape[0])
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[-1])
                        + ") and biases (" + Printable::toString(biasesShape[0]) + ")");
                    return Tensor();
                }

                return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

            }

        };

    }

}

#endif // BIAS_OP_HPP_