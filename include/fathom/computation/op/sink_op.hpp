#ifndef SINK_OP_HPP_
#define SINK_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class SinkOp : public virtual ComputationOp
    {
        
        DataContext::TensorIterator input_;

    public:

        SinkOp(OpType opType, const string &name);
        virtual ~SinkOp() = 0;
        virtual bool setInput(DataContext::TensorIterator &tensor, byte_type idx);
        virtual DataContext::TensorIterator getInput(byte_type idx);
        bool hasInputDef();
        byte_type inputSlots();

    };

}

#endif // SINK_OP_HPP_