#include "include/mcm/computation/op/computation_op.hpp"

mv::ComputationOp::ComputationOp(OpType opType, const std::string &name) :
ComputationElement(name)
{
    addAttr("opType", AttrType::OpTypeType, opType);
    logger_.log(Logger::MessageType::MessageDebug, "Defined computation op " + toString());
}

mv::ComputationOp::ComputationOp(mv::json::Value& value) :
ComputationElement(value)
{

}


mv::ComputationOp::~ComputationOp()
{

}

bool mv::ComputationOp::validOutputDef_()
{

    if (!hasInputDef())
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + "' because of undefined input/inputs");
        return false;
    }

    return true;

}

mv::OpType mv::ComputationOp::getOpType() const 
{
    return getAttr("opType").getContent<OpType>();
}

std::string mv::ComputationOp::toString() const
{
    return "op " + getAttr("opType").getContentStr() + " '" + name_ + "' " + ComputationElement::toString();
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
    return getAttr("executable").getContent<bool>();
}
