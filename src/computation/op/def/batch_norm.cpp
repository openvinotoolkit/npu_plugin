#include "include/mcm/computation/op/def/batch_norm.hpp"

mv::op::BatchNorm::BatchNorm(double varianceEps, const std::string &name) :
ComputationOp(OpType::BatchNorm, name),
SourceOp(OpType::BatchNorm, 1, name),
SinkOp(OpType::BatchNorm, 5, name)
{
    set<double>("varianceEps", varianceEps);
    set<bool>("executable", true);
}

mv::Tensor mv::op::BatchNorm::getOutputDef(std::size_t idx)
{
    
    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape(); 

    if (inputShape.ndims() != 3)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
            " has " + std::to_string(inputShape.ndims())));

    auto mean = getInputTensor(1);
    auto meanShape = mean->getShape();

    auto variance = getInputTensor(2);
    auto varianceShape = variance->getShape();

    auto offset = getInputTensor(3);
    auto offsetShape = offset->getShape();

    auto scale = getInputTensor(4);
    auto scaleShape = scale->getShape();

    if (meanShape != varianceShape || meanShape != offsetShape || meanShape != offsetShape)
        throw(OpError(*this, "Invalid dimensionality of parameter input tensors - must have an equal dimensionality, recevied"
            " mean (input 1) " + std::to_string(meanShape.ndims()) + ", variance (input 1) " + std::to_string(varianceShape.ndims()) +
            ", offset (input 3) " + std::to_string(offsetShape.ndims()) + ", scale (input 4) " + std::to_string(scaleShape.ndims())));

    if (inputShape != meanShape || inputShape != varianceShape || inputShape != offsetShape || inputShape != scaleShape)
    {

        if (meanShape.ndims() != 1)
            throw(OpError(*this, "Invalid shape of the mean tensor (input 1) - must have a dimensionality equal to 1 or"
                " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims())));
        
        if (meanShape[0] != inputShape[-1])
            throw(OpError(*this, "Invalid shape of the mean tensor (input 1) - if it has 1 dimension, it must be equal"
                " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1])));

        if (varianceShape.ndims() != 1)
            throw(OpError(*this, "Invalid shape of the variance tensor (input 1) - must have a dimensionality equal to 1 or"
                " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims())));
        
        if (varianceShape[0] != inputShape[-1])
            throw(OpError(*this, "Invalid shape of the variance tensor (input 1) - if it has 1 dimension, it must be equal"
                " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1])));
        
        if (offsetShape.ndims() != 1)
            throw(OpError(*this, "Invalid shape of the offset tensor (input 1) - must have a dimensionality equal to 1 or"
                " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims())));
        
        if (offsetShape[0] != inputShape[-1])
            throw(OpError(*this, "Invalid shape of the offset tensor (input 1) - if it has 1 dimension, it must be equal"
                " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1])));

        if (scaleShape.ndims() != 1)
            throw(OpError(*this, "Invalid shape of the scale tensor (input 1) - must have a dimensionality equal to 1 or"
                " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims())));
        
        if (scaleShape[0] != inputShape[-1])
            throw(OpError(*this, "Invalid shape of the scale tensor (input 1) - if it has 1 dimension, it must be equal"
                " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1])));

    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
    
}

bool mv::op::BatchNorm::isHardwarizeable(json::Object&)
{
    return false;
}
