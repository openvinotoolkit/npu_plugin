#include "include/mcm/computation/op/def/multiply.hpp"

mv::op::Multiply::Multiply(const string &name) :
ComputationOp(OpType::Multiply, name),
EltwiseOp(OpType::Multiply, name)
{
    addAttr("executable", AttrType::BoolType, true);
}