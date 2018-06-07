#ifndef ELTWISE_SUM_HPP_
#define ELTWISE_SUM_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    class EltwiseSum : public EltwiseOp
    {

    public:

        EltwiseSum(const Logger &logger, const string &opType, const UnpopulatedTensor &input0, const UnpopulatedTensor &input1, const string &name) :
        EltwiseOp(logger, "eltsum", input0, input1, name)
        {

        }

    };

}

#endif // ELTWISE_SUM_HPP_