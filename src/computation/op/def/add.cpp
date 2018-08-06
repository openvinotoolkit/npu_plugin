#include "include/mcm/computation/op/def/add.hpp"

mv::op::Add::Add(const string& name) :
ComputationOp(OpType::Add, name),
EltwiseOp(OpType::Add, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Add::Add(mv::json::Value& obj):
ComputationOp(obj),
EltwiseOp(obj)
{

}

bool mv::op::Add::isHardwarizeable(json::Object &TargetDescriptor)
{

}
