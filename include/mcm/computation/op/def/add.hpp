#ifndef ADD_HPP_
#define ADD_HPP_

#include "include/mcm/computation/op/eltwise_op.hpp"

namespace mv
{

    namespace op
    {

        class Add : public EltwiseOp
        {

        public:

            Add(const std::string& name);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;

        };

    }

}

#endif // ADD_HPP_
