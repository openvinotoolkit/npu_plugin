#include "include/fathom/computation/op/computation_op.hpp"

mv::ComputationOp::ComputationOp(const Logger &logger, const string &name, DType dType, Order order, Shape inputShape, Shape outputShape) :
ComputationElement(logger, "op_" + name),
dType_(dType),
order_(order),
inputShape_(inputShape),
outputShape_(outputShape)
{
    addAttr("dType", AttrType::DTypeType, dType_);
    addAttr("order", AttrType::OrderType, order_);
    addAttr("inputShape", AttrType::ShapeType, inputShape_);
    addAttr("outputShape", AttrType::ShapeType, outputShape_);
}

mv::ComputationOp::~ComputationOp()
{

}

/*mv::ComputationOp::ComputationOp(const ComputationOp &other) :
ComputationElement(other),
dType_(other.dType_),
order_(other.order_),
inputShape_(other.inputShape_),
outputShape_(other.outputShape_)
{

}*/

mv::Shape mv::ComputationOp::getInputShape() const
{
    return inputShape_;
}

mv::Shape mv::ComputationOp::getOutputShape() const
{
    return outputShape_;
}

mv::string mv::ComputationOp::toString() const
{
    return "'" + name_ + "' " + ComputationElement::toString();
}

mv::UnpopulatedModelTensor mv::ComputationOp::getOutput() const
{

    return UnpopulatedModelTensor(logger_, name_ + "_out:0", outputShape_, dType_, order_);

}
