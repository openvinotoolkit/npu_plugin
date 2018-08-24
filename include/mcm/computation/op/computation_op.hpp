#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/mcm/computation/model/computation_element.hpp"
#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/ops_register.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {

    protected:

        bool validOutputDef_();

    public:

        ComputationOp(OpType opType, const string& name);
        ComputationOp(mv::json::Value& value);
        virtual ~ComputationOp() = 0;

        OpType getOpType() const;
        string toString() const;

        virtual bool setInputTensor(Data::TensorIterator& tensor, byte_type idx);
        virtual bool setOutputTensor(Data::TensorIterator& tensor, byte_type idx);
        virtual Data::TensorIterator getInputTensor(byte_type idx);
        virtual Data::TensorIterator getOutputTensor(byte_type idx);
        virtual bool hasInputDef();
        virtual bool hasInputDef(byte_type idx);
        virtual Tensor getOutputDef(byte_type idx) = 0;
        virtual byte_type inputSlots();
        virtual byte_type outputSlots();
        bool isExecutable() const;
        virtual bool isHardwarizeable(mv::json::Object& TargetDescriptor) = 0;
        bool operator==(const ComputationOp &other) const;

    };

}

#endif // COMPUTATION_OP_HPP_
