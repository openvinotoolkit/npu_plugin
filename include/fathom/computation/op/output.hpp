#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include "include/fathom/computation/op/sink_op.hpp"

namespace mv
{

    class Output : public SinkOp
    {

    public:

        Output(const string &name) : 
        ComputationOp(OpType::Output, name),
        SinkOp(OpType::Output, 1, name)
        {
            addAttr("executable", AttrType::BoolType, false);
        }

        virtual bool setInput(DataContext::TensorIterator &tensor, byte_type idx)
        {

            bool result = SinkOp::setInput(tensor, idx);
            if (result)
            {
                addAttr("shape", AttrType::ShapeType, tensor->getShape());
                addAttr("dType", AttrType::DTypeType, tensor->getDType());
                addAttr("order", AttrType::OrderType, tensor->getOrder());        
            }
            return result;

        }

        Tensor getOutputDef(byte_type)
        {
            logger_.log(Logger::MessageType::MessageWarning, "Attempt of getting output tensor of model output operation");
            return Tensor();
        }

    };

}

#endif // OUTPUT_HPP_