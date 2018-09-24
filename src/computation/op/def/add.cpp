#include "include/mcm/computation/op/def/add.hpp"

mv::op::Add::Add(const std::string& name) :
ComputationOp(OpType::Add, name),
EltwiseOp(OpType::Add, name)
{
    set<bool>("executable", true);
}

bool mv::op::Add::isHardwarizeable(json::Object&)
{
    return false;
}
