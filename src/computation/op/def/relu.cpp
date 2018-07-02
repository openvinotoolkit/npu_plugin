#include "include/mcm/computation/op/def/relu.hpp"

mv::op::ReLU::ReLU(const string &name) :
ComputationOp(OpType::ReLU, name),
ActivationOp(OpType::ReLU, name)
{
    addAttr("executable", AttrType::BoolType, true);
}