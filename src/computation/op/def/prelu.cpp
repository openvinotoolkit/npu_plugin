#include "include/mcm/computation/op/def/prelu.hpp"

mv::op::PReLU::PReLU(const std::string &name) :
ComputationOp(OpType::PReLU, name),
SourceOp(OpType::PReLU, 1, name),
SinkOp(OpType::PReLU, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::PReLU::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto slope = getInputTensor(1);
    auto slopeShape = slope->getShape();

    if (slopeShape.ndims() != 1)
        throw(OpError(*this, "Incorrect shape " + slopeShape.toString() + " of slope (must be a vector)"));

    if (inputShape[-1] != slopeShape[0])
        throw(OpError(*this, "Mismatch in channels dimensions between input (" + std::to_string(inputShape[-1])
            + ") and slope (" + std::to_string(slopeShape[0]) + ")"));

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());

}

bool mv::op::PReLU::isHardwarizeable(json::Object &)
{
    return false;
}

void mv::op::PReLU::gatherSerialFields(){
    this->set<unsigned>("serialID", 10);
}