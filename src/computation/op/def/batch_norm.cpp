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

    auto mean = getInputTensor(1);
    auto meanShape = mean->getShape();

    auto variance = getInputTensor(2);
    auto varianceShape = variance->getShape();

    auto offset = getInputTensor(3);
    auto offsetShape = offset->getShape();

    auto scale = getInputTensor(4);
    auto scaleShape = scale->getShape();

    if (!(inputShape == meanShape && inputShape == varianceShape && inputShape == offsetShape && inputShape == scaleShape))
    {

        if (meanShape.ndims() != 1  || (meanShape[0] != inputShape[-1]))
            throw(OpError(*this, "Invalid shape of mean tensor (input 1) - has to be one dimensional tensor of"
                "dimension equal to " + std::to_string(inputShape[-1])));

        if (varianceShape.ndims() != 1  || (varianceShape[0] != inputShape[-1]))
            throw(OpError(*this, "Invalid shape of variance tensor (input 2) - has to be one dimensional tensor of"
                "dimension equal to " + std::to_string(inputShape[-1])));

        if (offsetShape.ndims() != 1  || (offsetShape[0] != inputShape[-1]))
            throw(OpError(*this, "Invalid shape of offset tensor (input 3) - has to be one dimensional tensor of"
                "dimension equal to " + std::to_string(inputShape[-1])));

         if (scaleShape.ndims() != 1  || (scaleShape[0] != inputShape[-1]))
            throw(OpError(*this, "Invalid shape of scale tensor (input 4) - has to be one dimensional tensor of"
                "dimension equal to " + std::to_string(inputShape[-1])));

    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
    
}

bool mv::op::BatchNorm::isHardwarizeable(json::Object&)
{
    return false;
}
