#include "include/mcm/computation/op/def/concat.hpp"

mv::op::Concat::Concat(const string &name) :
ComputationOp(OpType::Concat, name),
SinkOp(OpType::Concat, 2, name),
SourceOp(OpType::Concat, 1, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Concat::Concat(mv::json::Value& obj) :
ComputationOp(obj),
SinkOp(obj),
SourceOp(obj)
{

}

mv::Tensor mv::op::Concat::getOutputDef(byte_type idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape();
    if (input0Shape.ndims() != 3)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of incorrect shape " + input0Shape.toString() + " of input 0");
        return Tensor();
    }

    dim_type lastDim = input0Shape[2];

    for (byte_type i = 1; i < inputSlots(); ++i)
    {
        auto inputShape = getInputTensor(i)->getShape();
        if (inputShape.ndims() != 3)
        {
            logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape " + inputShape.toString() + " of input " + Printable::toString(i));
            return Tensor();
        }
        if (inputShape[0] != input0Shape[0] || inputShape[1] != input0Shape[1])
        {
            logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of inconsistent inputs shapes " + input0Shape.toString() + " and " + inputShape.toString());
            return Tensor();
        }

        lastDim += inputShape[2];

    }

    return Tensor(name_ + ":0", Shape(input0Shape[0], input0Shape[1], lastDim), input0->getDType(), input0->getOrder());
    
}
