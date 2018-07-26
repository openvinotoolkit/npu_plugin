#ifndef RELU_HPP_
#define RELU_HPP_

#include "include/mcm/computation/op/activation_op.hpp"

namespace mv
{

    namespace op
    {

        class ReLU : public ActivationOp
        {

        public:

            ReLU(const string &name);
            ReLU(mv::json::Value &obj);

        };

    }

}

#endif // RELU_HPP_
