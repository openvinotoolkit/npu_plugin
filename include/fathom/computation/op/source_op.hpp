#ifndef SOURCE_OP_HPP_
#define SOURCE_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class SourceOp : public virtual ComputationOp
    {
        
        DataContext::TensorIterator output_;

    public:

        SourceOp(OpType opType, const string &name);
        virtual ~SourceOp() = 0;
        virtual bool setOutput(DataContext::TensorIterator &tensor);
        virtual DataContext::TensorIterator getOutput();
        byte_type outputSlots();
        string getOutputName() const;

    };

}

#endif // SOURCE_OP_HPP_