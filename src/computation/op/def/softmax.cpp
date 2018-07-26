#include "include/mcm/computation/op/def/softmax.hpp"

mv::op::Softmax::Softmax(const string &name) :
ComputationOp(OpType::Softmax, name),
ActivationOp(OpType::Softmax, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Softmax::Softmax(mv::json::Value& obj) :
ComputationOp(obj),
ActivationOp(obj)
{

}
