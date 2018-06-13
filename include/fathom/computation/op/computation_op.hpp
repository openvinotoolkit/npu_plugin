#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"
#include "include/fathom/computation/op/ops_register.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"

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

        virtual bool setInput(Data::TensorIterator& tensor, byte_type idx);
        virtual bool setOutput(Data::TensorIterator& tensor, byte_type idx);
        virtual Data::TensorIterator getInput(byte_type idx);
        virtual Data::TensorIterator getOutput(byte_type idx);
        virtual bool hasInputDef();
        virtual bool hasInputDef(byte_type idx);
        virtual Tensor getOutputDef(byte_type idx) = 0;
        virtual byte_type inputSlots();
        bool isExecutable() const;

        bool operator==(const ComputationOp &other) const;

    };

}

#endif // COMPUTATION_OP_HPP_