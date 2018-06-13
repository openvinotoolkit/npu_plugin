#include "include/fathom/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(OpType opType, byte_type outputsCount, const string &name) :
ComputationOp(opType, name),
outputs_(outputsCount, Data::TensorIterator())
{
    addAttr("outputs", AttrType::ByteType, outputsCount);
}

mv::SourceOp::~SourceOp()
{

}

bool mv::SourceOp::setOutput(Data::TensorIterator &tensor, byte_type idx)
{
    
    if (idx >= getAttr("outputs").getContent<byte_type>())
        return false;   
    
    outputs_[idx] = tensor;
    addAttr("output" + Printable::toString(idx), AttrType::StringType, tensor->getName());
    logger_.log(Logger::MessageType::MessageDebug, "Set output " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;

}

mv::Data::TensorIterator mv::SourceOp::getOutput(byte_type idx)
{

    if (idx >= getAttr("outputs").getContent<byte_type>())
        return Data::TensorIterator();

    return outputs_[idx];

}

mv::byte_type mv::SourceOp::outputSlots()
{

    return outputs_.size();
    
}