#ifndef RELU_HPP_
#define RELU_HPP_

#include "include/mcm/computation/op/activation_op.hpp"

namespace mv
{

    namespace Op
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

}

#endif // RELU_HPP_