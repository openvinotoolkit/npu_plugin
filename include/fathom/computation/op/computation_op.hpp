#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/unpopulated.hpp"

namespace mv
{

    class ComputationOp : public ComputationElement
    {

        static allocator::map<string, size_type> idDict_;

    protected:

        string opType;
        DType dType_;
        Order order_;
        Shape inputShape_;
        Shape outputShape_;

    public:

        ComputationOp(const Logger &logger, const string &opType, DType dType, Order order, Shape inputShape, Shape outputShape, const string &name);
        virtual ~ComputationOp() = 0;

        DType getDType() const;
        Order getOrder() const;

        Shape getInputShape() const;
        Shape getOutputShape() const;

        UnpopulatedTensor getOutput() const;

        virtual string toString() const;

        bool operator==(const ComputationOp &other) const;

    };

}

#endif // COMPUTATION_OP_HPP_