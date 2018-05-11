#ifndef CONV_HPP_
#define CONV_HPP_

#include "include/fathom/computation/op/computation_op.hpp"
#include "include/fathom/computation/tensor/model_populated.hpp"

namespace mv
{

    class Conv : public ComputationOp
    {

    public:

        Conv(const Logger &logger, const string &name, const UnpopulatedModelTensor &input, const ConstantTensor &weights, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY) :
        ComputationOp(logger, "conv_" + name, input.getDType(), input.getOrder(), input.getShape(), Shape(input.getShape()[0], input.getShape()[1] / strideX, input.getShape()[2] / strideY, weights.getShape()[2]))
        {
            //addAttr("weights", AttrType::TensorType, PopulatedModelTensor(logger, name + "_weights", weights));
            addAttr("weights", AttrType::TensorType, weights);
            addAttr("strideX", AttrType::ByteType, strideX);
            addAttr("strideY", AttrType::ByteType, strideY);
            addAttr("padX", AttrType::ByteType, padX);
            addAttr("padY", AttrType::ByteType, padY);
        }

        string toString() const
        {
            return "conv " + ComputationOp::toString();
        }

    };

}

#endif // CONV_HPP_