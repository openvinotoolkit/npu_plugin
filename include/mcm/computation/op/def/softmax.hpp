#ifndef SOFTMAX_HPP_
#define SOFTMAX_HPP_

#include "include/mcm/computation/op/activation_op.hpp"

namespace mv
{

    namespace op
    {

        class Softmax : public ActivationOp
        {

        public:

            Softmax(const string &name);
            Softmax(mv::json::Value &obj);

        };

    }

}

#endif // SOFTMAX_HPP_
