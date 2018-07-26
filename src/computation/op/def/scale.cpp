#include "include/mcm/computation/op/def/scale.hpp"

mv::op::Scale::Scale(const string &name) :
ComputationOp(OpType::Scale, name),
SourceOp(OpType::Scale, 1, name),
SinkOp(OpType::Scale, 2, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Scale::Scale(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
SinkOp(obj)
{

}

mv::Tensor mv::op::Scale::getOutputDef(byte_type idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape(); 

    auto scale = getInputTensor(1);
    auto scaleShape = scale->getShape();

    if (inputShape != scaleShape)
    {

        if (scaleShape.ndims() != 1 || scaleShape[0] != inputShape[-1])
        {
            logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape of scale (" + scaleShape.toString() + ") - it needs to be either"
                " equal to shape of the input (" + inputShape.toString() + ") or to be one dimensional tensors of dimension " +
                Printable::toString(inputShape[-1]));
            return Tensor();
        }

    }

    return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
    
}
