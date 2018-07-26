#ifndef KERNEL_OP_HPP_
#define KERNEL_OP_HPP_

#include <cmath>
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{
    /// \todo Add assertions (dimensions)   
    class KernelOp : public SourceOp
    {

    protected:

        static dim_type getOutputDim_(dim_type inputDim, dim_type kernelDim, byte_type stride)
        {
            return (inputDim - kernelDim) / stride + 1;
        }

    public:

        KernelOp(OpType opType, UnsignedVector2D stride, UnsignedVector4D padding, const string &name);
        KernelOp(mv::json::Value& value);
        virtual ~KernelOp() = 0;

    };

}

#endif
