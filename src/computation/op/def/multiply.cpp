#include "include/mcm/computation/op/def/multiply.hpp"

mv::op::Multiply::Multiply(const string &name) :
ComputationOp(OpType::Muliply, name),
EltwiseOp(OpType::Muliply, name)
{
    addAttr("executable", AttrType::BoolType, true);
}