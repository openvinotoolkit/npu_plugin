#ifndef ELTWISE_MUL_HPP_
#define ELTWISE_MUL_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    class EltwiseMul : public EltwiseOp
    {

    public:

        EltwiseMul(const Logger &logger, const string &opType, const UnpopulatedTensor &input0, const UnpopulatedTensor &input1, const string &name) :
        EltwiseOp(logger, "eltmul", input0, input1, name)
        {

        }

    };

}

#endif // ELTWISE_MUL_HPP_