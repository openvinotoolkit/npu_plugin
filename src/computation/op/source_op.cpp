#include "include/mcm/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(OpType opType, std::size_t outputsCount, const std::string &name) :
ComputationOp(opType, name)
{
    set<unsigned short>("outputs", outputsCount);
}

mv::SourceOp::~SourceOp()
{

}

void mv::SourceOp::setOutputTensor(Data::TensorIterator tensor, std::size_t idx)
{
    
    if (idx >= get<unsigned short>("outputs"))
        throw IndexError(*this, idx, "Attempt of setting an undefined output");
    
    auto result = outputs_.emplace(idx, tensor);
    if (!result.second)
        outputs_[idx] = tensor;
    set<std::string>("output" + std::to_string(idx), tensor->getName());
    log(Logger::MessageType::Debug, "Set output " + std::to_string(idx) + " for " + toString() + " as " + tensor->toString());

}

mv::Data::TensorIterator mv::SourceOp::getOutputTensor(std::size_t idx)
{

    if (outputs_.find(idx) == outputs_.end())
        throw IndexError(*this, idx, "Attempt of getting an undefined output");

    return outputs_.at(idx);

}

std::size_t mv::SourceOp::outputSlots()
{

    return get<unsigned short>("outputs");
    
}
