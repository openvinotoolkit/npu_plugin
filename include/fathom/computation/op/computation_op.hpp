#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"
#include "include/fathom/computation/model/iterator/tensor_context.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {

        static allocator::map<string, size_type> idDict_;

    protected:

        bool validOutputDef_();

    public:

        ComputationOp(const Logger &logger, const string &opType, const string &name);
        virtual ~ComputationOp() = 0;

        string getOpType() const;
        string toString() const;
        virtual string getOutputName() const;

        virtual bool setInput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx);
        virtual bool setOutput(TensorContext::UnpopulatedTensorIterator &tensor);
        virtual TensorContext::UnpopulatedTensorIterator getInput(byte_type idx);
        virtual TensorContext::UnpopulatedTensorIterator getOutput();
        virtual bool hasInputDef();
        virtual UnpopulatedTensor getOutputDef() = 0;
        virtual byte_type inputSlots();

        bool operator==(const ComputationOp &other) const;

    };

}

#endif // COMPUTATION_OP_HPP_