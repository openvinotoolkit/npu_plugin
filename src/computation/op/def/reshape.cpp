#include "include/mcm/computation/op/def/reshape.hpp"

mv::op::Reshape::Reshape(Shape outputShape, const std::string& name) :
ComputationOp(OpType::Reshape, name),
SourceOp(OpType::Reshape, 1, name),
SinkOp(OpType::Reshape, 1, name)
{
    set<Shape>("shape", outputShape);
    set<bool>("executable", true);
}

mv::Tensor mv::op::Reshape::getOutputDef(std::size_t idx)
{
    
    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto outputShape = get<Shape>("shape");

    if (inputShape.totalSize() != outputShape.totalSize())
        throw(OpError(*this, "Invalid conversino of the original shape " + inputShape.toString() + " and the output shape "
            + outputShape.toString() + " - must have equal total number of elements"));

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}

bool mv::op::Reshape::isHardwarizeable(json::Object&)
{
    return false;
}
