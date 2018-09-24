#include "include/mcm/computation/op/def/relu.hpp"

mv::op::ReLU::ReLU(const std::string &name) :
ComputationOp(OpType::ReLU, name),
ActivationOp(OpType::ReLU, name)
{
    set<bool>("executable", true);
}

bool mv::op::ReLU::isHardwarizeable(json::Object&)
{
    return false;
}
