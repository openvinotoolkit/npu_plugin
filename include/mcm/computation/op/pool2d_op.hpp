#ifndef POOL2D_HPP_
#define POOL2D_HPP_

#include "include/mcm/computation/op/kernel_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)
    class Pool2DOp : public KernelOp, public SinkOp
    {

    public:

        Pool2DOp(OpType poolType, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const std::string &name);
        Pool2DOp(mv::json::Value& value);

        virtual ~Pool2DOp() = 0;
        Tensor getOutputDef(std::size_t idx);

    };
}

#endif // POOL2D_HPP_
