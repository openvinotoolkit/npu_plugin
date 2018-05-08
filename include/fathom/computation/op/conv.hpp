#ifndef CONV_HPP_
#define CONV_HPP_

#include "include/fathom/computation/op/computation_op.hpp"
#include "include/fathom/computation/tensor/model_constant.hpp"

namespace mv
{

    class Conv : public ComputationOp
    {

        ConstantModelTensor weights_;
        byte_type strideX_;
        byte_type strideY_;
        
    public:

        Conv(const Logger &logger, const string &name, const VariableTensor &input, const ConstantTensor &weights, byte_type strideX, byte_type strideY) :
        ComputationOp(logger, "conv_" + name, input.getDType(), input.getOrder(), input.getShape(), Shape(input.getShape()[0], input.getShape()[1] / strideX, input.getShape()[2] / strideY, weights.getShape()[2])),
        weights_(logger, name + "_weights", weights),
        strideX_(strideX),
        strideY_(strideY)
        {
            addAttr("weights", AttrType::TensorType, weights_);
            addAttr("strideX", AttrType::ByteType, strideX_);
            addAttr("strideY", AttrType::ByteType, strideY_);
        }

        string toString() const
        {
            return "conv " + ComputationOp::toString();
        }

    };

}

#endif // CONV_HPP_