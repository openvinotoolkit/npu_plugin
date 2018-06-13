#ifndef SOURCE_OP_HPP_
#define SOURCE_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class SourceOp : public virtual ComputationOp
    {
        
        dynamic_vector<Data::TensorIterator> outputs_;

    public:

        SourceOp(OpType opType, byte_type outputsCount, const string &name);
        virtual ~SourceOp() = 0;
        virtual bool setOutput(Data::TensorIterator &tensor, byte_type idx);
        virtual Data::TensorIterator getOutput(byte_type idx);
        byte_type outputSlots();

    };

}

#endif // SOURCE_OP_HPP_