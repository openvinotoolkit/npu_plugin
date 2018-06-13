#ifndef SOFTMAX_HPP_
#define SOFTMAX_HPP_

#include "include/fathom/computation/op/activation_op.hpp"

namespace mv
{

    namespace Op
    {

        class Softmax : public ActivationOp
        {

        public:

            Softmax(const string &name) :
            ComputationOp(OpType::Softmax, name),
            ActivationOp(OpType::Softmax, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

        };

    }

}

#endif // SOFTMAX_HPP_