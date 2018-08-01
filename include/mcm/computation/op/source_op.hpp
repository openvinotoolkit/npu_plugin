#ifndef SOURCE_OP_HPP_
#define SOURCE_OP_HPP_

#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class SourceOp : public virtual ComputationOp
    {
        
        dynamic_vector<Data::TensorIterator> outputs_;

    public:

        SourceOp(OpType opType, byte_type outputsCount, const string &name);
        SourceOp(mv::json::Value& value);
        virtual ~SourceOp() = 0;
        virtual bool setOutputTensor(Data::TensorIterator &tensor, byte_type idx);
        virtual Data::TensorIterator getOutputTensor(byte_type idx);
        byte_type outputSlots();

    };

}

#endif // SOURCE_OP_HPP_
