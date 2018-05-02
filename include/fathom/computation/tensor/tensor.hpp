#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "include/fathom/computation/tensor/shape.hpp"

namespace mv
{

    class Tensor
    {
    
    public:

        enum Order
        {
            NWHC
        };

        enum DType
        {
            Float
        };

    protected:

        Shape shape_;
        DType dType_;
        Order order_;

    public:

        Tensor(const Shape &shape, DType dType, Order order) : 
        shape_(shape),
        dType_(dType),
        order_(order)
        {

        }

        Shape getShape() const
        {
            return shape_;
        }

        DType getDType() const
        {
            return dType_;
        }

        Order getOrder() const
        {
            return order_;
        }
 
    };

}

#endif // MODEL_TENSOR_HPP_