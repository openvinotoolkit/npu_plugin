#include "include/mcm/computation/op/def/multiply.hpp"

mv::op::Multiply::Multiply(const std::string &name) :
ComputationOp(OpType::Multiply, name),
EltwiseOp(OpType::Multiply, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Multiply::Multiply(mv::json::Value& obj) :
ComputationOp(obj),
EltwiseOp(obj)
{

}

bool mv::op::Multiply::isHardwarizeable(json::Object&)
{
    return false;
}
