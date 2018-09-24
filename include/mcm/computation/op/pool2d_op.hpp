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

        Pool2DOp(OpType poolType, std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name);
        virtual ~Pool2DOp() = 0;
        Tensor getOutputDef(std::size_t idx);

    };
}

#endif // POOL2D_HPP_
