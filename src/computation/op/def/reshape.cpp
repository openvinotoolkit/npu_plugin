#include "include/mcm/computation/op/def/reshape.hpp"

mv::op::Reshape::Reshape(Shape outputShape, const string& name) :
ComputationOp(OpType::Reshape, name),
SourceOp(OpType::Reshape, 1, name),
SinkOp(OpType::Reshape, 1, name)
{
    addAttr("shape", AttrType::ShapeType, outputShape);
    addAttr("executable", AttrType::BoolType, true);
}

mv::Tensor mv::op::Reshape::getOutputDef(byte_type idx)
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
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because conversion of input shape " + inputShape.toString() + " and requested shape " + outputShape.toString() +
            " is impossible");
        return Tensor();
    }

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}
