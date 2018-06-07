#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class MultiSinkOp : public virtual ComputationOp
    {
        
        vector<TensorContext::UnpopulatedTensorIterator> inputs_;

    public:

        MultiSinkOp(const Logger &logger, const string &opType, byte_type inputsCount, const string &name);
        virtual ~MultiSinkOp() = 0;
        virtual bool setInput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx);
        virtual TensorContext::UnpopulatedTensorIterator getInput(byte_type idx);
        bool hasInputDef();
        byte_type inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_