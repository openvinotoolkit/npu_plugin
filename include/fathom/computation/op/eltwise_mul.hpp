#ifndef ELTWISE_MUL_HPP_
#define ELTWISE_MUL_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    class EltwiseMul : public EltwiseOp
    {

    public:

        EltwiseMul(const string &opType, const Tensor &input0, const Tensor &input1, const string &name) :
        EltwiseOp("eltmul", input0, input1, name)
        {

        }

    };

}

#endif // ELTWISE_MUL_HPP_