#include "include/fathom/computation/op/multisink_op.hpp"

mv::MultiSinkOp::MultiSinkOp(OpType opType, byte_type inputsCount, const string &name) :
ComputationOp(opType, name),
inputs_(inputsCount, TensorContext::TensorIterator())
{
    addAttr("inputs", AttrType::ByteType, inputsCount);
}

mv::MultiSinkOp::~MultiSinkOp()
{

}

bool mv::MultiSinkOp::setInput(TensorContext::TensorIterator &tensor, byte_type idx)
{
    if (idx >= getAttr("inputs").getContent<byte_type>())
        return false;

    inputs_[idx] = tensor;
    addAttr("input" + Printable::toString(idx), AttrType::StringType, tensor->getName());
    logger_.log(Logger::MessageType::MessageDebug, "Set input " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::TensorContext::TensorIterator mv::MultiSinkOp::getInput(byte_type idx)
{
     if (idx >= getAttr("inputs").getContent<byte_type>())
        return TensorContext::TensorIterator();

    return inputs_[idx];
}

bool mv::MultiSinkOp::hasInputDef()
{

    

    for (unsigned_type i = 0; i < inputs_.size(); ++i)
    {
        if (!hasInputDef(i))
            return false;
    }

    return true;

}

bool mv::MultiSinkOp::hasInputDef(byte_type idx)
{
    TensorContext::TensorIterator emptyIt;

    if (inputs_[idx] == emptyIt)
        return false;

    return true;
}

mv::byte_type mv::MultiSinkOp::inputSlots()
{
    return inputs_.size();
}
