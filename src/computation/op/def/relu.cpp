#include "include/mcm/computation/op/def/relu.hpp"

mv::op::ReLU::ReLU(const std::string &name) :
ComputationOp(OpType::ReLU, name),
ActivationOp(OpType::ReLU, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::ReLU::ReLU(mv::json::Value& obj) :
ComputationOp(obj),
ActivationOp(obj)
{

}

bool mv::op::ReLU::isHardwarizeable(json::Object&)
{
    return false;
}
