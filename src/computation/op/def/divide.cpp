#include "include/mcm/computation/op/def/divide.hpp"

mv::op::Divide::Divide(const std::string &name) :
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

bool mv::op::Divide::isHardwarizeable(json::Object&)
{
    return false;
}
