#include "include/mcm/computation/op/sink_op.hpp"

mv::SinkOp::SinkOp(OpType opType, std::size_t inputsCount, const std::string &name) :
ComputationOp(opType, name),
inputs_(inputsCount, Data::TensorIterator())
{
    addAttr("inputs", AttrType::ByteType, inputsCount);
}

mv::SinkOp::SinkOp(mv::json::Value& value) :
ComputationOp(value),
inputs_(getAttr("inputs").getContent<std::size_t>(), Data::TensorIterator())
{
    //Tensors cannot be filled here
}


mv::SinkOp::~SinkOp()
{

}

bool mv::SinkOp::setInputTensor(Data::TensorIterator &tensor, std::size_t idx)
{
    if (idx >= getAttr("inputs").getContent<std::size_t>())
        return false;

    inputs_[idx] = tensor;
    addAttr("input" + Printable::toString(idx), AttrType::StringType, tensor->getName());
    logger_.log(Logger::MessageType::MessageDebug, "Set input " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::Data::TensorIterator mv::SinkOp::getInputTensor(std::size_t idx)
{
    if (idx >= getAttr("inputs").getContent<std::size_t>())
        return Data::TensorIterator();

    return inputs_[idx];
}

bool mv::SinkOp::hasInputDef()
{

    for (std::size_t i = 0; i < inputs_.size(); ++i)
    {
        if (!hasInputDef(i))
            return false;
    }

    return true;

}

bool mv::SinkOp::hasInputDef(std::size_t idx)
{

    Data::TensorIterator emptyIt;

    if (inputs_[idx] == emptyIt)
        return false;

    return true;

}

std::size_t mv::SinkOp::inputSlots()
{

    return inputs_.size();
    
}
