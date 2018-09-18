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
        throw(OpError(*this, "Invalid shape of biases tensor (input 1) - has to be 1-dimensional, received "
            + std::to_string(biasesShape.ndims())));

    if (inputShape[-1] != biasesShape[0])
        throw(OpError(*this, "Invalid shape of biases tensor (input 1) - the dimension has to equal to the last dimension"
            " of the input tensor which is " + std::to_string(inputShape[-1])));

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

}

bool mv::op::Bias::isHardwarizeable(json::Object&)
{
    return false;
}

