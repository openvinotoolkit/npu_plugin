#include "include/mcm/computation/op/def/subtract.hpp"

mv::op::Subtract::Subtract(const std::string &name) :
ComputationOp(OpType::Subtract, name),
EltwiseOp(OpType::Subtract, name)
{
    set<bool>("executable", true);
}

/*mv::op::Subtract::Subtract(mv::json::Value& obj) :
ComputationOp(obj),
EltwiseOp(obj)
{

}*/

bool mv::op::Subtract::isHardwarizeable(json::Object&)
{
    return false;
}
