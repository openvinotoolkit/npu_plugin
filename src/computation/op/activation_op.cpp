#include "include/mcm/computation/op/activation_op.hpp"

mv::ActivationOp::ActivationOp(OpType activationType, const std::string& name) :
ComputationOp(activationType, name),
SourceOp(activationType, 1, name),
SinkOp(activationType, 1, name)
{

}

mv::ActivationOp::~ActivationOp()
{

}

mv::Tensor mv::ActivationOp::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);

    return Tensor(name_ + ":0", input->getShape(), input->getDType(), input->getOrder());

}
