#include "include/mcm/computation/op/sink_op.hpp"

mv::SinkOp::SinkOp(OpType opType, std::size_t inputsCount, const std::string &name) :
ComputationOp(opType, name)
{
    set<unsigned short>("inputs", inputsCount);
}

mv::SinkOp::~SinkOp()
{

}

bool mv::SinkOp::setInputTensor(Data::TensorIterator &tensor, std::size_t idx)
{
    if (idx >= get<unsigned short>("inputs"))
        return false;

    inputs_.emplace(idx, tensor);
    set<std::string>("input" + std::to_string(idx), tensor->getName());
    log(Logger::MessageType::MessageDebug, "Set input " + std::to_string(idx) + " for " + toString() + " as " + tensor->toString());
    return true;
}

mv::Data::TensorIterator mv::SinkOp::getInputTensor(std::size_t idx)
{
    if (idx >= get<unsigned short>("inputs"))
        throw IndexError(*this, idx, "Exceeds number of inputs");

    return inputs_.at(idx);
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

    if (inputs_.find(idx) == inputs_.end())
        return false;

    return true;

}

std::size_t mv::SinkOp::inputSlots()
{

    return get<unsigned short>("inputs");
    
}
