#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class MultiSinkOp : public virtual ComputationOp
    {
        
        dynamic_vector<DataContext::TensorIterator> inputs_;

    public:

        MultiSinkOp(OpType opType, byte_type inputsCount, const string &name);
        virtual ~MultiSinkOp() = 0;
        virtual bool setInput(DataContext::TensorIterator &tensor, byte_type idx);
        virtual DataContext::TensorIterator getInput(byte_type idx);
        bool hasInputDef();
        bool hasInputDef(byte_type idx);
        byte_type inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_