#include "include/fathom/computation/op/multisink_op.hpp"

mv::MultiSinkOp::MultiSinkOp(const Logger &logger, const string &opType, byte_type inputsCount, const string &name) :
ComputationOp(logger, opType, name),
inputs_(inputsCount, TensorContext::UnpopulatedTensorIterator())
{
    addAttr("inputs", AttrType::ByteType, inputsCount);
}

mv::MultiSinkOp::~MultiSinkOp()
{

}

bool mv::MultiSinkOp::setInput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx)
{
    if (idx >= getAttr("inputs").getContent<byte_type>())
        return false;

    inputs_[idx] = tensor;

    logger_.log(Logger::MessageType::MessageDebug, "Set input " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::TensorContext::UnpopulatedTensorIterator mv::MultiSinkOp::getInput(byte_type idx)
{
     if (idx >= getAttr("inputs").getContent<byte_type>())
        return TensorContext::UnpopulatedTensorIterator();

    return inputs_[idx];
}

bool mv::MultiSinkOp::hasInputDef()
{

    TensorContext::UnpopulatedTensorIterator emptyIt;

    for (unsigned_type i = 0; i < inputs_.size(); ++i)
    {
        if (inputs_[i] == emptyIt)
            return false;
    }

    return true;

}

mv::byte_type mv::MultiSinkOp::inputSlots()
{
    return inputs_.size();
}
