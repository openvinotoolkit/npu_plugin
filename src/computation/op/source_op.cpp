#include "include/mcm/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(OpType opType, std::size_t outputsCount, const std::string &name) :
ComputationOp(opType, name),
outputs_(outputsCount, Data::TensorIterator())
{
    addAttr("outputs", AttrType::ByteType, outputsCount);
}

mv::SourceOp::SourceOp(mv::json::Value& value) :
ComputationOp(value),
outputs_(getAttr("outputs").getContent<std::size_t>(), Data::TensorIterator())
{
    //Tensors cannot be filled here
}

mv::SourceOp::~SourceOp()
{

}

bool mv::SourceOp::setOutputTensor(Data::TensorIterator &tensor, std::size_t idx)
{
    
    if (idx >= getAttr("outputs").getContent<std::size_t>())
        return false;   
    
    outputs_[idx] = tensor;
    addAttr("output" + Printable::toString(idx), AttrType::StringType, tensor->getName());
    logger_.log(Logger::MessageType::MessageDebug, "Set output " + Printable::toString(idx) + " for " + toString() + " as " + tensor->toString());
    return true;

}

mv::Data::TensorIterator mv::SourceOp::getOutputTensor(std::size_t idx)
{

    if (idx >= getAttr("outputs").getContent<std::size_t>())
        return Data::TensorIterator();

    return outputs_[idx];

}

std::size_t mv::SourceOp::outputSlots()
{

    return outputs_.size();
    
}
