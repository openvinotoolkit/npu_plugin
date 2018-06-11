#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include "include/fathom/computation/op/multisink_op.hpp"

namespace mv
{

    class Output : public MultiSinkOp
    {

    public:

        Output(const string &name) : 
        ComputationOp(OpType::Output, name),
        MultiSinkOp(OpType::Output, 1, name)
        {
            addAttr("executable", AttrType::BoolType, false);
        }

        Tensor getOutputDef()
        {
            logger_.log(Logger::MessageType::MessageWarning, "Attempt of getting output tensor of model output operation");
            return Tensor();
        }

        virtual bool setInput(TensorContext::TensorIterator &tensor, byte_type idx)
        {

            bool result = MultiSinkOp::setInput(tensor, idx);
            if (result)
            {
                addAttr("shape", AttrType::ShapeType, tensor->getShape());
                addAttr("dType", AttrType::DTypeType, tensor->getDType());
                addAttr("order", AttrType::OrderType, tensor->getOrder());        
            }
            return result;

        }

    };

}

#endif // OUTPUT_HPP_