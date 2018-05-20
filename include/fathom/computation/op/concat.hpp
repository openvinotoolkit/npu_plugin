#ifndef CONCAT_HPP_
#define CONCAT_HPP_

#include "include/fathom/computation/op/computation_op.hpp"
#include "include/fathom/computation/tensor/populated.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)   
    class Concat : public ComputationOp
    {

    public:

        Concat(const Logger &logger, const UnpopulatedTensor &input0, const UnpopulatedTensor &input1, const string &name) :
        ComputationOp(logger, "concat", input0.getDType(), input0.getOrder(), input0.getShape(),
        Shape(input0.getShape()[0], input0.getShape()[1], input0.getShape()[2], input0.getShape()[3] + input1.getShape()[3]), name)
        {
            addAttr("input1Shape", AttrType::ShapeType, input1.getShape());
        }

        string toString() const
        {
            return "concat " + ComputationOp::toString();
        }

    };

}

#endif // CONCAT_HPP_