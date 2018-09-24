#include "include/mcm/computation/op/def/softmax.hpp"

mv::op::Softmax::Softmax(const std::string &name) :
ComputationOp(OpType::Softmax, name),
ActivationOp(OpType::Softmax, name)
{
    set<bool>("executable", true);
}

bool mv::op::Softmax::isHardwarizeable(json::Object&)
{
    return false;
}
