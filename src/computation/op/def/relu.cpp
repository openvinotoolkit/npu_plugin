#include "include/mcm/computation/op/def/relu.hpp"

mv::op::ReLu::ReLu(const string &name) :
ComputationOp(OpType::ReLu, name),
ActivationOp(OpType::ReLu, name)
{
    addAttr("executable", AttrType::BoolType, true);
}