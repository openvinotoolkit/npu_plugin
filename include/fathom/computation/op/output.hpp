#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include "include/fathom/computation/op/multisink_op.hpp"

namespace mv
{

    class Output : public MultiSinkOp
    {

    public:

        Output(const Logger &logger, const string &name) : 
        ComputationOp(logger, "output", name),
        MultiSinkOp(logger, "output", 1, name)
        {
            addAttr("executable", AttrType::BoolType, false);
        }

        UnpopulatedTensor getOutputDef()
        {
            logger_.log(Logger::MessageType::MessageWarning, "Attempt of getting output tensor of model output operation");
            return UnpopulatedTensor(logger_);
        }

        virtual bool setInput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx)
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