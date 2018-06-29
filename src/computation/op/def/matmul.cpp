#include "include/mcm/computation/op/def/matmul.hpp"

mv::op::MatMul::MatMul(const string &name) :
ComputationOp(OpType::MatMul, name),
SinkOp(OpType::MatMul, 2, name),
SourceOp(OpType::MatMul, 1, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::Tensor mv::op::MatMul::getOutputDef(byte_type idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape(); 
    auto input1Shape = getInputTensor(1)->getShape();

    if (input0Shape.ndims() != 2)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of incorrect shape " + input0Shape.toString() + " of input 0");
        return Tensor();
    }

    if (input1Shape.ndims() != 2)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of incorrect shape " + input1Shape.toString() + " of input 1");
        return Tensor();
    }

    if (input0Shape[1] != input1Shape[0])
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of inconsistent shape of input 0 " + input0Shape.toString() + " and input 1 " + input1Shape.toString());
        return Tensor();
    }

    return Tensor(name_ + ":0", Shape(input0Shape[0], input1Shape[1]), input0->getDType(), input0->getOrder());
    
}