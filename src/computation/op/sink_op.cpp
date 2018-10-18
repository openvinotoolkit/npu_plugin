#include "include/mcm/computation/op/sink_op.hpp"

mv::SinkOp::SinkOp(OpType opType, std::size_t inputsCount, const std::string &name) :
ComputationOp(opType, name)
{
    set<unsigned short>("inputs", inputsCount);
}

mv::SinkOp::~SinkOp()
{

}

void mv::SinkOp::setInputTensor(Data::TensorIterator tensor, std::size_t idx)
{
    if (idx >= get<unsigned short>("inputs"))
        throw IndexError(*this, idx, "Attempt of setting an undefined input");

    auto result = inputs_.emplace(idx, tensor);
    if (!result.second)
        inputs_[idx] = tensor;
    set<std::string>("input" + std::to_string(idx), tensor->getName());
    log(Logger::MessageType::Debug, "Set input " + std::to_string(idx) + " for " + toString() + " as " + tensor->toString());
}

mv::Data::TensorIterator mv::SinkOp::getInputTensor(std::size_t idx)
{
    if (inputs_.find(idx) == inputs_.end())
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
