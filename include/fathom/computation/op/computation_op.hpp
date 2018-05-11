#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/model_unpopulated.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {

    protected:

        DType dType_;
        Order order_;
        Shape inputShape_;
        Shape outputShape_;

    public:

        ComputationOp(const Logger &logger, const string &name, DType dType, Order order, Shape inputShape, Shape outputShape);
        virtual ~ComputationOp() = 0;
        //ComputationOp(const ComputationOp &other);

        DType getDType() const;
        Order getOrder() const;

        Shape getInputShape() const;
        Shape getOutputShape() const;

        UnpopulatedModelTensor getOutput() const;

        virtual string toString() const;

    };

}

#endif // COMPUTATION_OP_HPP_