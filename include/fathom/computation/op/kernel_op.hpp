#ifndef KERNEL_OP_HPP_
#define KERNEL_OP_HPP_

#include <cmath>
#include "include/fathom/computation/op/source_op.hpp"


namespace mv
{
    /// \todo Add assertions (dimensions)   
    class KernelOp : public SourceOp
    {

    protected:

        static dim_type getOutputDim_(dim_type inputDim, dim_type kernelDim, byte_type padding, byte_type stride)
        {
            return (inputDim - kernelDim + 2 * padding) / stride + 1;
        }

    public:

        KernelOp(const string &opType, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name);
        virtual ~KernelOp() = 0;

    };

}

#endif