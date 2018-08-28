#ifndef CONV2D_OP_HPP_
#define CONV2D_OP_HPP_

#include "include/mcm/computation/op/kernel_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)   
        class Conv2D : public KernelOp, public SinkOp
        {

        public:

            Conv2D(UnsignedVector2D stride, UnsignedVector4D padding, const std::string& name);
            Conv2D(mv::json::Value &obj);

            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // CONV2D_OP_HPP_
