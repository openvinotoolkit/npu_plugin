#include "include/mcm/computation/op/def/input.hpp"

mv::op::Input::Input(Shape outputShape, DType dType, Order order, const std::string &name) :
ComputationOp(OpType::Input, name),
SourceOp(OpType::Input, 1, name)
{

    addAttr("shape", AttrType::ShapeType, outputShape);
    addAttr("dType", AttrType::DTypeType, dType);
    addAttr("order", AttrType::OrderType, order);
    addAttr("executable", AttrType::BoolType, false);

}

mv::op::Input::Input(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj)
{

}

bool mv::op::Input::setOutputTensor(Data::TensorIterator &tensor, std::size_t idx)
{

    bool result = SourceOp::setOutputTensor(tensor, idx);
    return result;

}

mv::Tensor mv::op::Input::getOutputDef(std::size_t idx)
{

    if (idx > 0)
        return Tensor();

    auto outputShape = getAttr("shape").getContent<Shape>();
    auto dType = getAttr("dType").getContent<DType>();
    auto order = getAttr("order").getContent<Order>();
    return Tensor(name_ + ":0", outputShape, dType, order);

}

bool mv::op::Input::isHardwarizeable(json::Object&)
{
    return false;
}
