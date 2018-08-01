#include "include/mcm/computation/op/def/subtract.hpp"

mv::op::Subtract::Subtract(const string &name) :
ComputationOp(OpType::Subtract, name),
EltwiseOp(OpType::Subtract, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Subtract::Subtract(mv::json::Value& obj) :
ComputationOp(obj),
EltwiseOp(obj)
{

}
