#include "include/mcm/computation/op/def/divide.hpp"

mv::op::Divide::Divide(const std::string &name) :
ComputationOp(OpType::Divide, name),
EltwiseOp(OpType::Divide, name)
{
    set<bool>("executable", true);
}

bool mv::op::Divide::isHardwarizeable(json::Object&)
{
    return false;
}
