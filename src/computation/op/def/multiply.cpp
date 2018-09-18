#include "include/mcm/computation/op/def/multiply.hpp"

mv::op::Multiply::Multiply(const std::string &name) :
ComputationOp(OpType::Multiply, name),
EltwiseOp(OpType::Multiply, name)
{
    set<bool>("executable", true);
}

bool mv::op::Multiply::isHardwarizeable(json::Object&)
{
    return false;
}
