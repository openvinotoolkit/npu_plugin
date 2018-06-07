#include "include/fathom/computation/op/sink_op.hpp"

mv::SinkOp::SinkOp(const Logger &logger, const string &opType, const string &name) :
ComputationOp(logger, opType, name)
{
    addAttr("inputs", AttrType::ByteType, (byte_type)1);
}

mv::SinkOp::~SinkOp()
{

}

bool mv::SinkOp::setInput(TensorContext::UnpopulatedTensorIterator &tensor, byte_type idx)
{
    if (idx > 1)
        return false;

    input_ = tensor;
    logger_.log(Logger::MessageType::MessageDebug, "Set input 0 for " + toString() + " as " + tensor->toString());
    return true;
}

mv::TensorContext::UnpopulatedTensorIterator mv::SinkOp::getInput(byte_type idx)
{
    if (idx > 1)
        return TensorContext::UnpopulatedTensorIterator();

    return input_;

}

bool mv::SinkOp::hasInputDef()
{

    TensorContext::UnpopulatedTensorIterator emptyIt;
    
    if (input_ == emptyIt)
        return false;
    
    return true;

}

mv::byte_type mv::SinkOp::inputSlots()
{
    return 1;
}