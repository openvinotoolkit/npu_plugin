#include "include/fathom/computation/op/op.hpp"

mv::ComputationOp::ComputationOp(const Logger &logger, const Shape &inputShape, const Shape &outputShape,
Tensor::DType dType, Tensor::Order order, const string &name) :
ComputationElement(logger, "Op_" + name),
inputShape_(inputShape),
outputShape_(outputShape),
dType_(dType),
order_(order)
{

}

mv::Shape mv::ComputationOp::getInputShape() const
{
    return inputShape_;
}

mv::Shape mv::ComputationOp::getOutputShape() const
{
    return outputShape_;
}

mv::Tensor::DType mv::ComputationOp::getDType() const
{
    return dType_; 
}

mv::Tensor::Order mv::ComputationOp::getOrder() const
{
    return order_;
}

mv::string mv::ComputationOp::toString() const
{
    auto inStr = inputShape_.toString();
    auto outStr = outputShape_.toString();
    return "'" + name_ + "' " + inStr + " -> " + outStr + " " + ComputationElement::toString() + " ";
}