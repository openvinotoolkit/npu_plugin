#include "include/mcm/computation/op/def/divide.hpp"

mv::op::Divide::Divide(const string &name) :
ComputationOp(OpType::Divide, name),
EltwiseOp(OpType::Divide, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Divide::Divide(mv::json::Value& obj) :
ComputationOp(obj),
EltwiseOp(obj)
{

}
