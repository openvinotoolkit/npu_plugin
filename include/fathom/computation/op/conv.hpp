#ifndef CONV_HPP_
#define CONV_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"
#include "include/fathom/computation/tensor/populated.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)   
    class Conv : public KernelOp
    {

    public:

        Conv(const Logger &logger, const UnpopulatedTensor &input, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
        KernelOp(logger, "conv", input, Shape(weights.getShape()[0], weights.getShape()[1], weights.getShape()[3]), strideX, strideY, padX, padY, name)
        {
            addAttr("weights", AttrType::TensorType, weights);
        }

        string toString() const
        {
            return "conv " + ComputationOp::toString();
        }

    };

}

#endif // CONV_HPP_