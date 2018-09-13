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

            Multiply(const std::string &name);
            Multiply(mv::json::Value &obj);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // MULTIPLY_HPP_
