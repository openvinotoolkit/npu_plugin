#include "include/mcm/computation/op/def/constant.hpp"

mv::op::Constant::Constant(const std::vector<double> &data, const Shape &shape, DType dType, Order order, const std::string &name) :
ComputationOp(OpType::Constant, name),
SourceOp(OpType::Constant, 1, name),
data_(data)
{
    set<Shape>("shape", shape);
    set<DType>("dType", dType);
    set<Order>("order", order);
    set<bool>("executable", false);
}

mv::Tensor mv::op::Constant::getOutputDef(std::size_t idx)
{
    
    // Will throw on error
    validOutputDef_(idx);

    auto shape = get<Shape>("shape");
    auto dType = get<DType>("dType");
    auto order = get<Order>("order");
    return Tensor(name_ + ":0", shape, dType, order, data_);
}

bool mv::op::Constant::isHardwarizeable(json::Object&)
{
    return false;
}
