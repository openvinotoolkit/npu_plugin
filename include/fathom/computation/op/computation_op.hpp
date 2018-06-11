#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"
#include "include/fathom/computation/model/iterator/tensor_context.hpp"
#include "include/fathom/computation/op/ops_register.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {

        static allocator::map<OpType, size_type> idDict_;

    protected:

        bool validOutputDef_();

    public:

        ComputationOp(OpType opType, const string& name);
        virtual ~ComputationOp() = 0;

        OpType getOpType() const;
        string toString() const;
        virtual string getOutputName() const;

        virtual bool setInput(TensorContext::TensorIterator& tensor, byte_type idx);
        virtual bool setOutput(TensorContext::TensorIterator& tensor);
        virtual TensorContext::TensorIterator getInput(byte_type idx);
        virtual TensorContext::TensorIterator getOutput();
        virtual bool hasInputDef();
        virtual bool hasInputDef(byte_type idx);
        virtual Tensor getOutputDef() = 0;
        virtual byte_type inputSlots();

        bool operator==(const ComputationOp &other) const;

    };

}

#endif // COMPUTATION_OP_HPP_