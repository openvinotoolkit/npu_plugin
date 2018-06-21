#ifndef RELU_HPP_
#define RELU_HPP_

#include "include/mcm/computation/op/activation_op.hpp"

namespace mv
{

    namespace op
    {

        class ReLu : public ActivationOp
        {

        public:

            ReLu(const string &name);

        };

    }

}

#endif // RELU_HPP_