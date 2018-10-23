#include "include/mcm/computation/op/def/matmul.hpp"

mv::op::MatMul::MatMul(const std::string &name) :
ComputationOp(OpType::MatMul, name),
SinkOp(OpType::MatMul, 2, name),
SourceOp(OpType::MatMul, 1, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::MatMul::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape();
    auto input1Shape = getInputTensor(1)->getShape();

    if (input0Shape.ndims() != 2)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 2, "
            " has " + std::to_string(input0Shape.ndims())));

    if (input0Shape.ndims() != 2)
        throw(OpError(*this, "Invalid shape of the parameters tensor (input 1) - must have a dimensionality of 2, "
            " has " + std::to_string(input0Shape.ndims())));

    if (input0Shape[1] != input1Shape[0])
        throw(OpError(*this, "Mismatch between the second dimensinon of the input tensor (input 0) " + std::to_string(input0Shape[1]) +
            " and the first dimension of the parameters tensor (input 1) " + std::to_string(input1Shape[0])));

    return Tensor(name_ + ":0", {input0Shape[0], input1Shape[1]}, input0->getDType(), input0->getOrder());

}

bool mv::op::MatMul::isHardwarizeable(json::Object&)
{
    return false;
}