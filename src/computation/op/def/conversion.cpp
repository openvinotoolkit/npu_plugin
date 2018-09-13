#include "include/mcm/computation/op/def/conversion.hpp"

mv::op::Conversion::Conversion(const std::string &name, mv::Order targetOrder):
ComputationOp(mv::OpType::Conversion, name),
SinkOp(mv::OpType::Conversion, 1, name),
SourceOp(mv::OpType::Conversion, 1, name)
{
    set<Order>("target_order", targetOrder);
    set<bool>("executable", true);
}

/*mv::op::Conversion::Conversion(mv::json::Value& value):
ComputationOp(value),
SinkOp(value),
SourceOp(value)
{

}*/

mv::Tensor mv::op::Conversion::getOutputDef(std::size_t idx)
{
    /*if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();*/

    auto input = getInputTensor(0);

    //Target order handled here
    return Tensor(name_ + ":0", input->getShape(), input->getDType(), get<Order>("target_order"));
}

bool mv::op::Conversion::isHardwarizeable(mv::json::Object&)
{
    return false;
}
