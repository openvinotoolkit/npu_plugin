#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include <map>
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class SinkOp : public virtual ComputationOp
    {

    protected:
        std::map<std::size_t, Data::TensorIterator> inputs_;

    public:

        SinkOp(OpType opType, std::size_t inputsCount, const std::string &name);
        virtual ~SinkOp() = 0;
        virtual void setInputTensor(Data::TensorIterator tensor, std::size_t idx) override;
        virtual Data::TensorIterator getInputTensor(std::size_t idx);
        virtual bool hasInputDef();
        virtual bool hasInputDef(std::size_t idx);
        std::size_t inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_
