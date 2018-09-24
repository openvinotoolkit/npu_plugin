#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class SinkOp : public virtual ComputationOp
    {
        
        std::vector<Data::TensorIterator> inputs_;

    public:

        SinkOp(OpType opType, std::size_t inputsCount, const std::string &name);
        virtual ~SinkOp() = 0;
        virtual bool setInputTensor(Data::TensorIterator &tensor, std::size_t idx);
        virtual Data::TensorIterator getInputTensor(std::size_t idx);
        bool hasInputDef();
        bool hasInputDef(std::size_t idx);
        std::size_t inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_
