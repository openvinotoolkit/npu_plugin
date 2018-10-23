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

            Subtract(const std::string &name);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;

        };

    }

}

#endif // SUBTRACT_HPP_
