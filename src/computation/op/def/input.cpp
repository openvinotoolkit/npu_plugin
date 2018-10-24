#include "include/mcm/computation/op/def/input.hpp"

mv::op::Input::Input(Shape outputShape, DType dType, Order order, const std::string &name) :
ComputationOp(OpType::Input, name),
SourceOp(OpType::Input, 1, name)
{

    set<Shape>("shape", outputShape);
    set<DType>("dType", dType);
    set<Order>("order", order);
    set<bool>("executable", false);

}

void mv::op::Input::setOutputTensor(Data::TensorIterator tensor, std::size_t idx)
{
    SourceOp::setOutputTensor(tensor, idx);
}

mv::Tensor mv::op::Input::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto outputShape = get<Shape>("shape");
    auto dType = get<DType>("dType");
    auto order = get<Order>("order");

    //Order should be a 2D order here
    return Tensor(name_ + ":0", outputShape, dType, order);

}

bool mv::op::Input::isHardwarizeable(json::Object&)
{
    return false;
}
