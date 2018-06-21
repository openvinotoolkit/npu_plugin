#include "include/mcm/computation/op/sink_op.hpp"

mv::SinkOp::SinkOp(OpType opType, byte_type inputsCount, const string &name) :
ComputationOp(opType, name),
inputs_(inputsCount, Data::TensorIterator())
{
    addAttr("inputs", AttrType::ByteType, inputsCount);
}

mv::SinkOp::~SinkOp()
{

}

bool mv::SinkOp::setInputTensor(Data::TensorIterator &tensor, byte_type idx)
{
    if (idx >= getAttr("inputs").getContent<byte_type>())
        return false;

    inputs_[idx] = tensor;
    addAttr("input" + Printable::toString(idx), AttrType::StringType, tensor->getName());
    logger_.log(Logger::MessageType::MessageDebug, "Set input " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::Data::TensorIterator mv::SinkOp::getInputTensor(byte_type idx)
{
    if (idx >= getAttr("inputs").getContent<byte_type>())
        return Data::TensorIterator();

    return inputs_[idx];
}

bool mv::SinkOp::hasInputDef()
{

    for (unsigned_type i = 0; i < inputs_.size(); ++i)
    {
        if (!hasInputDef(i))
            return false;
    }

    return true;

}

bool mv::SinkOp::hasInputDef(byte_type idx)
{

    Data::TensorIterator emptyIt;

    if (inputs_[idx] == emptyIt)
        return false;

    return true;

}

mv::byte_type mv::SinkOp::inputSlots()
{

    return inputs_.size();
    
}
