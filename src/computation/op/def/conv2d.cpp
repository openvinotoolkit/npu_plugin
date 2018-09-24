#include "include/mcm/computation/op/def/conv2d.hpp"

mv::op::Conv2D::Conv2D(std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name) :
ComputationOp(OpType::Conv2D, name),
KernelOp(OpType::Conv2D, stride, padding, name),
SinkOp(OpType::Conv2D, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::Conv2D::getOutputDef(std::size_t idx)
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

    if (inputShape[2] != weightsShape[2])
        throw(OpError(*this, "Mismatch in channels dimensions between input tensor (input 0) and weights tensor (input 1) - "
            " input" + std::to_string(inputShape[3]) + ", weights " + std::to_string(weightsShape[2])));
    
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

bool mv::op::Conv2D::isHardwarizeable(json::Object&)
{
    auto padding = get<std::array<unsigned short, 4>>("padding");
    auto stride = get<std::array<unsigned short, 2>>("stride");

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    // Check for supported padding
    if((padding[0] != 0 && padding[0] != weightsShape[0]/2) || (padding[2] != 0 && padding[2] != weightsShape[1]/2))
        return false;

    // Check for supported kernel sizes
    if(weightsShape[0] > 15 || weightsShape[1] > 15)
        return false;

    // Check for supported strides
    if(stride[0] > 8 || stride[1] > 8)
        return false;


    // Should handle dilation (WTF?) here

    return true;
}
