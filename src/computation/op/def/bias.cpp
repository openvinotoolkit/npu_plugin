#include "include/mcm/computation/op/def/bias.hpp"

mv::op::Bias::Bias(const std::string& name) :
ComputationOp(OpType::Bias, name),
SourceOp(OpType::Bias, 1, name),
SinkOp(OpType::Bias, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::Bias::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto biases = getInputTensor(1);
    auto biasesShape = biases->getShape();
    
    if (biasesShape.ndims() != 1)
        throw(OpError(*this, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape " + biasesShape.toString() + " of biases (must be a vector)"));

    if (inputShape[-1] != biasesShape[0])
    {
        log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of mismatch in channels dimensions between input (" + std::to_string(inputShape[-1])
            + ") and biases (" + std::to_string(biasesShape[0]) + ")");
        //return Tensor();
    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

}

bool mv::op::Bias::isHardwarizeable(json::Object&)
{
    return false;
}

