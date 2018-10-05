#ifndef SOURCE_OP_HPP_
#define SOURCE_OP_HPP_

#include <map>
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class SourceOp : public virtual ComputationOp
    {
        
        std::map<std::size_t, Data::TensorIterator> outputs_;

    public:

        SourceOp(OpType opType, std::size_t outputsCount, const std::string &name);
        virtual ~SourceOp() = 0;
        virtual bool setOutputTensor(Data::TensorIterator &tensor, std::size_t idx);
        virtual Data::TensorIterator getOutputTensor(std::size_t idx);
        std::size_t outputSlots();

    };

}

#endif // SOURCE_OP_HPP_
