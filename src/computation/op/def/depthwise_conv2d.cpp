#include "include/mcm/computation/op/def/depthwise_conv2d.hpp"

mv::op::DepthwiseConv2D::DepthwiseConv2D(std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name) :
ComputationOp(OpType::DepthwiseConv2D, name),
KernelOp(OpType::DepthwiseConv2D, stride, padding, name),
SinkOp(OpType::DepthwiseConv2D, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::DepthwiseConv2D::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    if (inputShape.ndims() != 3)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
            " has " + std::to_string(inputShape.ndims())));

    if (weightsShape.ndims() != 4)
        throw(OpError(*this, "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 4, "
            " has " + std::to_string(inputShape.ndims())));

    if (weightsShape[2] != 1)
        throw(OpError(*this, "Mismatch in channels dimensions of weights tensor (input 1) - "
             + std::to_string(weightsShape[2]) + " (expected 1)"));

    if (inputShape[2] != weightsShape[3])
        throw(OpError(*this, "Mismatch in channels dimensions in weights tensor (input 1) - Input channels "
            " input channels " + std::to_string(inputShape[2]) + " are different from output channels " + std::to_string(weightsShape[3])));

    auto padding = get<std::array<unsigned short, 4>>("padding");
    auto stride = get<std::array<unsigned short, 2>>("stride");

    if (inputShape[0] + padding[0] + padding[1] < weightsShape[0])
        throw(OpError(*this, "Filter kernel width (" + std::to_string(weightsShape[0]) + ") exceeds the padded input width ("
            + std::to_string(inputShape[0] + padding[0] + padding[1]) + ")"));

    if (inputShape[1] + padding[2] + padding[3] < weightsShape[1])
        throw(OpError(*this, "Filter kernel height (" + std::to_string(weightsShape[1]) + ") exceeds the padded input height ("
            + std::to_string(inputShape[1] + padding[2] + padding[3]) + ")"));

    // Make sure that the result of subtract will not be negative
    Shape outputShape({(inputShape[0] + padding[0] + padding[1] - weightsShape[0]) / stride[0] + 1, (
        inputShape[1] + padding[2] + padding[3] - weightsShape[1]) / stride[1] + 1, weightsShape[3]});

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}

bool mv::op::DepthwiseConv2D::isHardwarizeable(json::Object&)
{
   return false;
}
