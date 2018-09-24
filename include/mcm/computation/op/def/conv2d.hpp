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

            Conv2D(std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // CONV2D_OP_HPP_
