#include "include/fathom/computation/op/computation_op.hpp"

mv::allocator::map<mv::string, mv::size_type> mv::ComputationOp::idDict_(allocator_);

mv::ComputationOp::ComputationOp(const Logger &logger, const string &opType, DType dType, Order order, Shape inputShape, Shape outputShape, const string &name) :
ComputationElement(logger, "op_" + opType + "_" + [&name, &opType]() -> string { if (name.empty()) return Printable::toString(idDict_[opType]++); else return name; }()),
dType_(dType),
order_(order),
inputShape_(inputShape),
outputShape_(outputShape)
{
    logger_.log(Logger::MessageType::MessageDebug, "Defined computation op " + toString());
    addAttr("opType", AttrType::StringType, opType);
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

mv::UnpopulatedTensor mv::ComputationOp::getOutput() const
{

    return UnpopulatedTensor(logger_, name_ + "_out:0", outputShape_, dType_, order_);

}

bool mv::ComputationOp::operator==(const ComputationOp &other) const
{
    return getName() == other.getName();
}