#include "include/mcm/computation/op/computation_op.hpp"

mv::allocator::map<mv::OpType, mv::size_type> mv::ComputationOp::idDict_;

mv::ComputationOp::ComputationOp(OpType opType, const string &name) :
ComputationElement(Printable::toString(opType) + "_" + [&name, &opType]() -> string { if (name.empty()) return Printable::toString(idDict_[opType]++); else return name; }())
{
    logger_.log(Logger::MessageType::MessageDebug, "Defined computation op " + toString());
    addAttr("opType", AttrType::OpTypeType, opType);
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

mv::string mv::ComputationOp::toString() const
{
    return "op " + getAttr("opType").getContentStr() + " '" + name_ + "' " + ComputationElement::toString();
}

mv::Data::TensorIterator mv::ComputationOp::getInput(byte_type)
{
    return mv::Data::TensorIterator();
}

mv::Data::TensorIterator mv::ComputationOp::getOutput(byte_type)
{
    return mv::Data::TensorIterator();
}

bool mv::ComputationOp::setInput(Data::TensorIterator&, byte_type)
{
    return false;
}

bool mv::ComputationOp::setOutput(Data::TensorIterator&, byte_type)
{
    return false;
}

bool mv::ComputationOp::hasInputDef()
{
    return false;
}

bool mv::ComputationOp::hasInputDef(byte_type)
{
    return false;
}

mv::byte_type mv::ComputationOp::inputSlots()
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