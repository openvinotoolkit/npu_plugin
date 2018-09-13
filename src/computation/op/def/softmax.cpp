#include "include/mcm/computation/op/def/softmax.hpp"

mv::op::Softmax::Softmax(const std::string &name) :
ComputationOp(OpType::Softmax, name),
ActivationOp(OpType::Softmax, name)
{
    set<bool>("executable", true);
}

/*mv::op::Softmax::Softmax(mv::json::Value& obj) :
ComputationOp(obj),
ActivationOp(obj)
{

}*/

bool mv::op::Softmax::isHardwarizeable(json::Object&)
{
    return false;
}
