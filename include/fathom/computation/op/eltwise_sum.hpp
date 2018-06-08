#ifndef ELTWISE_SUM_HPP_
#define ELTWISE_SUM_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    class EltwiseSum : public EltwiseOp
    {

    public:

        EltwiseSum(const string &opType, const Tensor &input0, const Tensor &input1, const string &name) :
        EltwiseOp("eltsum", input0, input1, name)
        {

        }

    };

}

#endif // ELTWISE_SUM_HPP_