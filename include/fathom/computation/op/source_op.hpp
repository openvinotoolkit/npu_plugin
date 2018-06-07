#ifndef SOURCE_OP_HPP_
#define SOURCE_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class SourceOp : public virtual ComputationOp
    {
        
        TensorContext::UnpopulatedTensorIterator output_;

    public:

        SourceOp(const Logger &logger, const string &opType, const string &name);
        virtual ~SourceOp() = 0;
        virtual bool setOutput(TensorContext::UnpopulatedTensorIterator &tensor);
        virtual TensorContext::UnpopulatedTensorIterator getOutput();
        byte_type outputSlots();
        string getOutputName() const;

    };

}

#endif // SOURCE_OP_HPP_