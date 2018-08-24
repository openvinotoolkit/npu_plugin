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

            Add(const string& name);
            Add(mv::json::Value& obj);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };
        
    }

}

#endif // ADD_HPP_
