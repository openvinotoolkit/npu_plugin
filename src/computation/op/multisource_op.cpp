/*#include "include/fathom/computation/op/multisource_op.hpp"

mv::MultiSourceOp::MultiSourceOp(const Logger &logger, const string &opType, byte_type outputsCount, const string &name) :
ComputationOp(logger, opType, name),
outputs_(outputsCount, TensorContext::UnpopulatedTensorIterator())
{
    addAttr("outputs", AttrType::ByteType, outputsCount);
}

mv::MultiSourceOp::~MultiSourceOp()
{

}

bool mv::MultiSourceOp::setOutput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx)
{
    if (idx >= getAttr("outputs").getContent<byte_type>())
        return false;

    outputs_[idx] = tensor;

    logger_.log(Logger::MessageType::MessageDebug, "Set output " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::TensorContext::UnpopulatedTensorIterator mv::MultiSourceOp::getOutput(byte_type idx)
{
    if (idx >= getAttr("outputs").getContent<byte_type>())
        return TensorContext::UnpopulatedTensorIterator();

    return outputs_[idx];
}

mv::byte_type mv::MultiSourceOp::outputSlots()
{
    return outputs_.size();
}*/