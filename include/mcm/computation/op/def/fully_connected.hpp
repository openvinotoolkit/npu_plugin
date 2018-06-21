#ifndef FULLY_CONNECTED_HPP_
#define FULLY_CONNECTED_HPP_

#include "include/mcm/computation/op/sink_op.hpp"
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{

    namespace Op
    {

        /// \todo Add assertions (dimensions)   
        class FullyConnected : public SinkOp, public SourceOp
        {

        public:

            FullyConnected(const string &name) :
            ComputationOp(OpType::FullyConnected, name),
            SinkOp(OpType::FullyConnected, 2, name),
            SourceOp(OpType::FullyConnected, 1, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

            Tensor getOutputDef(byte_type idx)
            {
                
                if (idx > 0)
                    return Tensor();

                if (!validOutputDef_())
                    return Tensor();

                auto input0 = getInputTensor(0);
                auto input0Shape = input0->getShape(); 
                auto input1Shape = getInputTensor(1)->getShape();

                if (input0Shape.ndims() != 2)
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + input0Shape.toString() + " of input 0");
                    return Tensor();
                }

                if (input1Shape.ndims() != 2)
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + input1Shape.toString() + " of input 1");
                    return Tensor();
                }

                if (input1Shape[1] != input0Shape[0])
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of inconsistent shape of input 0 " + input0Shape.toString() + " and input 1 " + input1Shape.toString());
                    return Tensor();
                }

                return Tensor(name_ + ":0", Shape(input1Shape[0], input0Shape[1]), input0->getDType(), input0->getOrder());
                
            }

        };

    }

}

#endif // FULLY_CONNECTED_HPP_