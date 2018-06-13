#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class SinkOp : public virtual ComputationOp
    {
        
        dynamic_vector<Data::TensorIterator> inputs_;

    public:

        SinkOp(OpType opType, byte_type inputsCount, const string &name);
        virtual ~SinkOp() = 0;
        virtual bool setInput(Data::TensorIterator &tensor, byte_type idx);
        virtual Data::TensorIterator getInput(byte_type idx);
        bool hasInputDef();
        bool hasInputDef(byte_type idx);
        byte_type inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_