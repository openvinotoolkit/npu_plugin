#ifndef MULTIPLY_HPP_
#define MULTIPLY_HPP_

#include "include/fathom/computation/op/eltwise_op.hpp"

namespace mv
{

    class Multiply : public EltwiseOp
    {

    public:

        Multiply(const string &name) :
        ComputationOp(OpType::Muliply, name),
        EltwiseOp(OpType::Muliply, name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

    };

}

#endif // MULTIPLY_HPP_