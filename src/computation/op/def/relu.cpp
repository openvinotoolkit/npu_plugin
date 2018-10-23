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

void mv::op::ReLU::gatherSerialFields(){
    this->set<unsigned>("opX", 0);
    this->set<unsigned>("strideX", 0);
    this->set<unsigned>("strideY", 0);
}