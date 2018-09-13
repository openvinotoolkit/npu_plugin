#include "include/mcm/computation/op/computation_op.hpp"

mv::ComputationOp::ComputationOp(OpType opType, const std::string &name) :
Element(name),
opType_(opType)
{
    log(Logger::MessageType::MessageDebug, "Defined");
    //addAttr("opType", mv::AttrType::OpTypeType, opType);   
}

mv::ComputationOp::~ComputationOp()
{

}

void mv::ComputationOp::validOutputDef_(std::size_t idx)
{

    if (!hasInputDef())
        throw(OpError(*this, "Unable to determine the output tensor while inputs are undefined"));

    if (idx > 0)
        throw(IndexError(*this, idx, "Exceeds the number of outputs defined for this op type"));

}

mv::OpType mv::ComputationOp::getOpType() const 
{
    return opType_;
}

std::string mv::ComputationOp::toString() const
{
    return "op " + getOpType().toString() + " '" + name_ + "' " + Element::attrsToString_();
}

mv::Data::TensorIterator mv::ComputationOp::getInputTensor(std::size_t)
{
    return mv::Data::TensorIterator();
}

mv::Data::TensorIterator mv::ComputationOp::getOutputTensor(std::size_t)
{
    return mv::Data::TensorIterator();
}

bool mv::ComputationOp::setInputTensor(Data::TensorIterator&, std::size_t)
{
    return false;
}

bool mv::ComputationOp::setOutputTensor(Data::TensorIterator&, std::size_t)
{
    return false;
}

bool mv::ComputationOp::hasInputDef()
{
    return false;
}

bool mv::ComputationOp::hasInputDef(std::size_t)
{
    return false;
}

std::size_t mv::ComputationOp::inputSlots()
{
    return 0;
}

std::size_t mv::ComputationOp::outputSlots()
{
    return 0;
}

bool mv::ComputationOp::operator==(const ComputationOp &other) const
{
    return getName() == other.getName();
}

bool mv::ComputationOp::isExecutable() const
{
    return get<bool>("executable");
}

std::string mv::ComputationOp::getLogID() const
{
    return "Op " + getOpType().toString() + " '" + getName() + "'";
}