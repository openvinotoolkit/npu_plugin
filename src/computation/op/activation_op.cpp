#include "include/mcm/computation/op/activation_op.hpp"

mv::ActivationOp::ActivationOp(OpType activationType, const string& name) :
ComputationOp(activationType, name),
SourceOp(activationType, 1, name),
SinkOp(activationType, 1, name)
{

}

mv::ActivationOp::~ActivationOp()
{

}

mv::Tensor mv::ActivationOp::getOutputDef(byte_type idx)
{

    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInput(0);

    return Tensor(name_ + ":0", input->getShape(), input->getDType(), input->getOrder());

}