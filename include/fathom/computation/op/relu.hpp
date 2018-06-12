#ifndef RELU_HPP_
#define RELU_HPP_

#include "include/fathom/computation/op/activation_op.hpp"

namespace mv
{

    class ReLu : public ActivationOp
    {

    public:

        ReLu(const string &name) :
        ComputationOp(OpType::ReLu, name),
        ActivationOp(OpType::ReLu, name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

    };

}

#endif // RELU_HPP_