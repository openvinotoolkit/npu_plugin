#ifndef MULTIPLY_HPP_
#define MULTIPLY_HPP_

#include "include/mcm/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace op
    {

        class Multiply : public EltwiseOp
        {

        public:

            Multiply(const string &name);

        };

    }

}

#endif // MULTIPLY_HPP_