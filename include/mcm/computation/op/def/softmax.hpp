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

            Softmax(const std::string &name);
            Softmax(mv::json::Value &obj);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // SOFTMAX_HPP_
