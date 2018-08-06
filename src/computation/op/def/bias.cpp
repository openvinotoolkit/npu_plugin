#include "include/mcm/computation/op/def/bias.hpp"

mv::op::Bias::Bias(const string& name) :
ComputationOp(OpType::Bias, name),
SourceOp(OpType::Bias, 1, name),
SinkOp(OpType::Bias, 2, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Bias::Bias(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
SinkOp(obj)
{

}

mv::Tensor mv::op::Bias::getOutputDef(byte_type idx)
{

    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto biases = getInputTensor(1);
    auto biasesShape = biases->getShape();
    
    if (biasesShape.ndims() != 1)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape " + biasesShape.toString() + " of biases (must be a vector)");
        return Tensor();
    }

    if (inputShape[-1] != biasesShape[0])
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[-1])
            + ") and biases (" + Printable::toString(biasesShape[0]) + ")");
        return Tensor();
    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

}

bool mv::op::Bias::isHardwarizeable(json::Object &TargetDescriptor)
{
    return false;
}

