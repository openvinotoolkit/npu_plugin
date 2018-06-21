#include "include/mcm/computation/op/def/softmax.hpp"

mv::op::Softmax::Softmax(const string &name) :
ComputationOp(OpType::Softmax, name),
ActivationOp(OpType::Softmax, name)
{
    addAttr("executable", AttrType::BoolType, true);
}
