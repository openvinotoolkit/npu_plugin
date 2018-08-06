#ifndef SUBTRACT_HPP_
#define SUBTRACT_HPP_

#include "include/mcm/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace op
    {

        class Subtract : public EltwiseOp
        {

        public:

            Subtract(const string &name);
            Subtract(mv::json::Value &obj);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // SUBTRACT_HPP_
