#include "include/mcm/computation/op/def/prelu.hpp"

mv::op::PReLU::PReLU(const string &name) :
ComputationOp(OpType::PReLU, name),
SourceOp(OpType::PReLU, 1, name),
SinkOp(OpType::PReLU, 2, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::PReLU::PReLU(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
SinkOp(obj)
{

}


mv::Tensor mv::op::PReLU::getOutputDef(byte_type idx)
{

    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto slope = getInputTensor(1);
    auto slopeShape = slope->getShape();

    if (slopeShape.ndims() != 1)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ +
                "' because of incorrect shape " + slopeShape.toString() + " of slope (must be a vector)");
        return Tensor();
    }

    if (inputShape[-1] != slopeShape[0])
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ +
            "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[-1])
            + ") and slope (" + Printable::toString(slopeShape[0]) + ")");
        return Tensor();
    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

}

bool mv::op::PReLU::isHardwarizeable(json::Object &TargetDescriptor)
{
    return false;
}
