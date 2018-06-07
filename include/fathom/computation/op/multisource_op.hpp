/*#ifndef MULTISOURCE_OP_HPP_
#define MULTISOURCE_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class MultiSourceOp : public virtual ComputationOp
    {
        
        vector<TensorContext::UnpopulatedTensorIterator> outputs_;

    public:

        MultiSourceOp(const Logger &logger, const string &opType, byte_type outputsCount, const string &name);
        virtual ~MultiSourceOp() = 0;
        virtual bool setOutput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx);
        virtual TensorContext::UnpopulatedTensorIterator getOutput(byte_type idx);
        byte_type outputSlots();

    };

}

#endif // MULTISOURCE_OP_HPP_*/