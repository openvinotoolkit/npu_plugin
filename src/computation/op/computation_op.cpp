#include "include/mcm/computation/op/computation_op.hpp"

mv::ComputationOp::ComputationOp(OpType opType, const std::string &name) :
Element(name)
{
    set<OpType>("opType", opType); 
    log(Logger::MessageType::MessageInfo, "Initialized");     
}

mv::ComputationOp::~ComputationOp()
{
    log(Logger::MessageType::MessageInfo, "Deleted");
}

void mv::ComputationOp::validOutputDef_(std::size_t idx)
{

    if (!hasInputDef())
        throw(OpError(*this, "Unable to determine the output tensor while inputs are undefined"));

    if (idx > outputSlots())
        throw(IndexError(*this, idx, "Exceeds the number of outputs defined for this op type"));

}

mv::OpType mv::ComputationOp::getOpType() const 
{
    return get<OpType>("opType");
}

std::string mv::ComputationOp::toString() const
{
    return "Op " + getOpType().toString() + " '" + name_ + "' " + Element::attrsToString_();
}

mv::Data::TensorIterator mv::ComputationOp::getInputTensor(std::size_t idx)
{
    throw IndexError(*this, idx, "Attempt of getting an undefined input");
}

mv::Data::TensorIterator mv::ComputationOp::getOutputTensor(std::size_t idx)
{
    throw IndexError(*this, idx, "Attempt of getting an undefined output");
}

void mv::ComputationOp::setInputTensor(Data::TensorIterator, std::size_t idx)
{
    throw IndexError(*this, idx, "Attempt of setting an undefined input");
}

void mv::ComputationOp::setOutputTensor(Data::TensorIterator, std::size_t idx)
{
    throw IndexError(*this, idx, "Attempt of setting an undefined output");
}

bool mv::ComputationOp::hasInputDef()
{
    return true;
}

bool mv::ComputationOp::hasInputDef(std::size_t)
{
    return true;
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
    return "Op '" + getName() + "'";
}