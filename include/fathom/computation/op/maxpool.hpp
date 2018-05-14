#ifndef MAXPOOL_HPP_
#define MAXPOOL_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)
    class MaxPool : public KernelOp
    {

    public:

        MaxPool(const Logger &logger, const UnpopulatedTensor &input, Shape kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
        KernelOp(logger, "maxpool", input, Shape(kernelShape[0], kernelShape[1], input.getShape()[3]), strideX, strideY, padX, padY, name)
        {
            addAttr("kSize", AttrType::ShapeType, kernelShape);
        }

        string toString() const
        {
            return "maxpool " + ComputationOp::toString();
        }

    };

}

#endif // MAXPOOL_HPP_