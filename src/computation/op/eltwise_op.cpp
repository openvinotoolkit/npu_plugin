#include "include/mcm/computation/op/eltwise_op.hpp"

mv::EltwiseOp::EltwiseOp(OpType eltwiseType, const std::string &name) :
ComputationOp(eltwiseType, name),
SourceOp(eltwiseType, 1, name),
SinkOp(eltwiseType, 2, name)
{

}

mv::EltwiseOp::~EltwiseOp()
{

}

mv::Tensor mv::EltwiseOp::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape();
    auto input1Shape = getInputTensor(1)->getShape();

    if (input0Shape != input1Shape)
        throw(OpError(*this, "Invalid shape of input tensors - must be an equal, recevied "
            + input0Shape.toString() + " and " + input1Shape.toString()));

    return Tensor(name_ + ":0", input0Shape, input0->getDType(), input0->getOrder());

}
