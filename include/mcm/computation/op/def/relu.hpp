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

            ReLU(const std::string &name);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;
        };

    }

}

#endif // RELU_HPP_
