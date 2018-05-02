#include "include/fathom/computation/tensor/tensor.hpp"

mv::Tensor::Tensor(const Shape &shape, DType dType, Order order) : 
shape_(shape),
dType_(dType),
order_(order)
{

}

mv::Tensor::~Tensor()
{
    
}

mv::Shape mv::Tensor::getShape() const
{
    return shape_;
}

mv::Tensor::DType mv::Tensor::getDType() const
{
    return dType_;
}

mv::Tensor::Order mv::Tensor::getOrder() const
{
    return order_;
}