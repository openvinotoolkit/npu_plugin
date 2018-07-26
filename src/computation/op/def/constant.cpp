#include "include/mcm/computation/op/def/constant.hpp"

mv::op::Constant::Constant(const dynamic_vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name) :
ComputationOp(OpType::Constant, name),
SourceOp(OpType::Constant, 1, name),
data_(data)
{
    addAttr("shape", AttrType::ShapeType, shape);
    addAttr("dType", AttrType::DTypeType, dType);
    addAttr("order", AttrType::OrderType, order);
    addAttr("executable", AttrType::BoolType, false);
}

mv::json::Value mv::op::Constant::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationOp::toJsonValue();
    //toReturn["data"] = mv::Jsonable::toJsonValue(data_);
    return toReturn;
}

mv::op::Constant::Constant(mv::json::Value& obj) :
ComputationOp(obj),
SourceOp(obj),
data_(mv::Jsonable::constructFloatVectorFromJson(obj["data"]))
{

}

mv::Tensor mv::op::Constant::getOutputDef(byte_type idx)
{
    
    if (idx > 0)
        return Tensor();

    auto shape = getAttr("shape").getContent<Shape>();
    auto dType = getAttr("dType").getContent<DType>();
    auto order = getAttr("order").getContent<Order>();
    return Tensor(name_ + ":0", shape, dType, order, data_);
}
