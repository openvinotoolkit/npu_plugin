#include "include/mcm/computation/op/def/subtract.hpp"

mv::op::Subtract::Subtract(const std::string &name) :
ComputationOp(OpType::Subtract, name),
EltwiseOp(OpType::Subtract, name)
{
    set<bool>("executable", true);
}

bool mv::op::Subtract::isHardwarizeable(json::Object&)
{
    return false;
}

void mv::op::Subtract::gatherSerialFields(){
    this->set<unsigned>("SerialID", 12);
}