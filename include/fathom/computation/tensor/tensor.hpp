#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "include/fathom/computation/tensor/shape.hpp"

namespace mv
{
    
    class Tensor
    {

    protected:

        Shape shape_;
        DType dType_;
        Order order_;

    public:

        Tensor(const Shape &shape, DType dType, Order order);
        virtual ~Tensor() = 0;

        Tensor(const Tensor &other);
        
        Shape getShape() const;
        DType getDType() const;
        Order getOrder() const;

    };

}

#endif // MODEL_TENSOR_HPP_