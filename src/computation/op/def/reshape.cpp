#include "include/mcm/computation/op/def/reshape.hpp"

mv::op::Reshape::Reshape(Shape outputShape, const std::string& name) :
ComputationOp(OpType::Reshape, name),
SourceOp(OpType::Reshape, 1, name),
SinkOp(OpType::Reshape, 1, name)
{
    addAttr("shape", AttrType::ShapeType, outputShape);
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Reshape::Reshape(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
SinkOp(obj)
{

}

mv::Tensor mv::op::Reshape::getOutputDef(std::size_t idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto outputShape = getAttr("shape").getContent<Shape>();

    if (inputShape.totalSize() != outputShape.totalSize())
    {
        log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because conversion of input shape " + inputShape.toString() + " and requested shape " + outputShape.toString() +
            " is impossible");
        return Tensor();
    }

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}

bool mv::op::Reshape::isHardwarizeable(json::Object&)
{
    return false;
}
