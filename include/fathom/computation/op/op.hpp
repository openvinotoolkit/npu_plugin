#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {
        
        Shape inputShape_;
        Shape outputShape_;
        Tensor::DType dType_;
        Tensor::Order order_;

    public:

        ComputationOp(const Logger &logger, const Shape &inputShape, const Shape &outputShape,
        Tensor::DType dType, Tensor::Order order, const string &name);

        Shape getInputShape() const;
        Shape getOutputShape() const;
        Tensor::DType getDType() const;
        Tensor::Order getOrder() const;

        string toString() const;

    };

}

#endif // COMPUTATION_OP_HPP_