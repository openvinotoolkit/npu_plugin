#include "include/mcm/computation/op/def/batch_norm.hpp"

mv::op::BatchNorm::BatchNorm(double varianceEps, const std::string &name) :
ComputationOp(OpType::BatchNorm, name),
SourceOp(OpType::BatchNorm, 1, name),
SinkOp(OpType::BatchNorm, 5, name)
{
    addAttr("varianceEps", AttrType::FloatType, varianceEps);
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::BatchNorm::BatchNorm(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
SinkOp(obj)
{

}

mv::Tensor mv::op::BatchNorm::getOutputDef(std::size_t idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

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

        if ((meanShape.ndims() != 1 || varianceShape.ndims() != 1 || offsetShape.ndims() != 1 || scaleShape.ndims() != 1) ||
            (meanShape[0] != inputShape[-1] || varianceShape[0] != inputShape[-1] || offsetShape[0] != inputShape[-1] || scaleShape[0] != inputShape[-1]))
        {
            logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape of mean (" + meanShape.toString() + ") or variance (" + varianceShape.toString() +
                ") or offset (" + offsetShape.toString() + ") or scale (" + scaleShape.toString() + ") - they need to be either"
                " equal to shape of the input (" + inputShape.toString() + ") or to be one dimensional tensors of dimension " +
                Printable::toString(inputShape[-1]));
            return Tensor();
        }

    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
    
}

bool mv::op::BatchNorm::isHardwarizeable(json::Object&)
{
    return false;
}
