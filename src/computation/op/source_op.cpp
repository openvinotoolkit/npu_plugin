#include "include/mcm/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(OpType opType, std::size_t outputsCount, const std::string &name) :
ComputationOp(opType, name),
outputs_(outputsCount, Data::TensorIterator())
{
    set<unsigned short>("outputs", outputsCount);
}

mv::SourceOp::~SourceOp()
{

}

bool mv::SourceOp::setOutputTensor(Data::TensorIterator &tensor, std::size_t idx)
{
    
    if (idx >= get<unsigned short>("outputs"))
        return false;   
    
    outputs_[idx] = tensor;
    set<std::string>("output" + std::to_string(idx), tensor->getName());
    log(Logger::MessageType::MessageDebug, "Set output " + std::to_string(idx) + " for " + toString() + " as " + tensor->toString());
    return true;

}

mv::Data::TensorIterator mv::SourceOp::getOutputTensor(std::size_t idx)
{

    if (idx >= get<unsigned short>("outputs"))
        return Data::TensorIterator();

    return outputs_[idx];

}

std::size_t mv::SourceOp::outputSlots()
{

    return outputs_.size();
    
}
