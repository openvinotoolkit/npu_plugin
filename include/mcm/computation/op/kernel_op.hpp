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

        static std::size_t getOutputDim_(std::size_t inputDim, std::size_t kernelDim, std::size_t stride)
        {
            return (inputDim - kernelDim) / stride + 1;
        }

    public:

        KernelOp(OpType opType, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name);
        KernelOp(json::Value& value);
        virtual ~KernelOp() = 0;

    };

}

#endif
