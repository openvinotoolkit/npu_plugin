#ifndef ELTWISE_OP_HPP_
#define ELTWISE_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class EltwiseOp : public ComputationOp
    {

    public:

        EltwiseOp(const Logger &logger, const string &opType, const UnpopulatedTensor &input0, const UnpopulatedTensor &input1, const string &name);
        virtual ~EltwiseOp() = 0;

    };

}

#endif // ELTWISE_OP_HPP_