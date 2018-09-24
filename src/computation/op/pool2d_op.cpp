#include "include/mcm/computation/op/pool2d_op.hpp"

mv::Pool2DOp::Pool2DOp(OpType poolType, std::array<unsigned short, 2> kernelSize, 
    std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name) :
ComputationOp(poolType, name),
KernelOp(poolType, stride, padding, name),
SinkOp(poolType, 1, name)
{
    set<std::array<short unsigned, 2>>("kSize", kernelSize);
}

mv::Pool2DOp::~Pool2DOp()
{

}


mv::Tensor mv::Pool2DOp::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();

    if (inputShape.ndims() != 3)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
            " has " + std::to_string(inputShape.ndims())));

    auto padding = get<std::array<unsigned short, 4>>("padding");
    auto stride = get<std::array<unsigned short, 2>>("stride");
    auto kSize = get<std::array<unsigned short, 2>>("kSize");
    
    if (inputShape[0] + padding[0] + padding[1] < kSize[0])
        throw(OpError(*this, "Filter kernel width (" + std::to_string(kSize[0]) + ") exceeds the padded input width ("
            + std::to_string(inputShape[0] + padding[0] + padding[1]) + ")"));

    if (inputShape[1] + padding[2] + padding[3] < kSize[1])
        throw(OpError(*this, "Filter kernel height (" + std::to_string(kSize[1]) + ") exceeds the padded input height ("
            + std::to_string(inputShape[1] + padding[2] + padding[3]) + ")"));

    Shape outputShape({(inputShape[0] + padding[0] + padding[1] - kSize[0]) / stride[0] + 1,
        (inputShape[1] + padding[2] + padding[3] - kSize[1]) / stride[1] + 1, inputShape[2]});

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}
