#include "include/mcm/computation/op/def/scale.hpp"

mv::op::Scale::Scale(const std::string &name) :
ComputationOp(OpType::Scale, name),
SourceOp(OpType::Scale, 1, name),
SinkOp(OpType::Scale, 2, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::Scale::getOutputDef(std::size_t idx)
{
    
    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);
    auto inputShape = input->getShape(); 

    auto scale = getInputTensor(1);
    auto scaleShape = scale->getShape();

    if (inputShape != scaleShape)
    {

        if (scaleShape.ndims() != 1)
            throw(OpError(*this, "Invalid shape of the scale tensor (input 1) - must have a dimensionality equal to 1 or"
                " to dimensionality of the input tensor (tensor 0) which is " + std::to_string(inputShape.ndims())));
        
        if (scaleShape[0] != inputShape[-1])
            throw(OpError(*this, "Invalid shape of the scale tensor (input 1) - if it has 1 dimension, it must be equal"
                " to the last dimension of the input tensor (tensor 0) which is " + std::to_string(inputShape[-1])));

    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
    
}

bool mv::op::Scale::isHardwarizeable(json::Object&)
{
    return false;
}
