#ifndef KERNEL_OP_HPP_
#define KERNEL_OP_HPP_

#include <cmath>
#include "include/fathom/computation/op/computation_op.hpp"
#include "include/fathom/computation/tensor/populated.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)   
    class KernelOp : public ComputationOp
    {

    protected:

        static dim_type getOutputDim(dim_type inputDim, dim_type kernelDim, byte_type padding, byte_type stride)
        {
            return (inputDim - kernelDim + 2 * padding) / stride + 1;
        }

    public:

        KernelOp(const Logger &logger, const string &opType, const UnpopulatedTensor &input, Shape kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name);
        virtual ~KernelOp() = 0;
        string toString() const;

    };

}

#endif