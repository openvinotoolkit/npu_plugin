#include "include/mcm/computation/op/def/fully_connected.hpp"

mv::op::FullyConnected::FullyConnected(const string &name) :
ComputationOp(OpType::FullyConnected, name),
SinkOp(OpType::FullyConnected, 2, name),
SourceOp(OpType::FullyConnected, 1, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::FullyConnected::FullyConnected(mv::json::Value& obj) :
ComputationOp(obj),
SinkOp(obj),
SourceOp(obj)
{

}

mv::Tensor mv::op::FullyConnected::getOutputDef(byte_type idx)
{
    
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape(); 
    auto input1Shape = getInputTensor(1)->getShape();

    if (input1Shape.ndims() != 2)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of incorrect shape " + input1Shape.toString() + " of weights matrix");
        return Tensor();
    }

    if (input0Shape.totalSize() != input1Shape[0])
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of inconsistent total size of input " + Printable::toString(input0Shape.totalSize()) + 
            " and 1st dimension of weights matrix " + Printable::toString(input1Shape[0]));
        return Tensor();
    }

    return Tensor(name_ + ":0", Shape(1, input1Shape[1]), input0->getDType(), input0->getOrder());
    
}

bool mv::op::FullyConnected::isHardwarizeable(json::Object &TargetDescriptor)
{
    return false;
}
