#include "include/fathom/computation/tensor/tensor.hpp"

mv::Tensor::Tensor(const Shape &shape, DType dType, Order order) : 
shape_(shape),
dType_(dType),
order_(order)
{

}

mv::Tensor::Tensor(const Tensor &other) :
shape_(other.shape_),
dType_(other.dType_),
order_(other.order_)
{

}

mv::Tensor::~Tensor()
{
    
}

mv::Shape mv::Tensor::getShape() const
{
    return shape_;
}

mv::DType mv::Tensor::getDType() const
{
    return dType_;
}

mv::Order mv::Tensor::getOrder() const
{
    return order_;
}