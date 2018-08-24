#include "include/mcm/computation/op/def/conversion.hpp"

mv::op::Conversion::Conversion(const string &name, mv::Order targetOrder):
ComputationOp(mv::OpType::Conversion, name),
SinkOp(mv::OpType::Conversion, 1, name),
SourceOp(mv::OpType::Conversion, 1, name)
{
    addAttr("target_order", mv::Attribute(mv::AttrType::OrderType, targetOrder));
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Conversion::Conversion(mv::json::Value& value):
ComputationOp(value),
SinkOp(value),
SourceOp(value)
{

}

mv::Tensor mv::op::Conversion::getOutputDef(byte_type idx)
{
    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);

    //Target order handled here
    return Tensor(name_ + ":0", input->getShape(), input->getDType(), getAttr("target_order").getContent<mv::Order>());
}

bool mv::op::Conversion::isHardwarizeable(mv::json::Object& TargetDescriptor)
{
    return false;
}
