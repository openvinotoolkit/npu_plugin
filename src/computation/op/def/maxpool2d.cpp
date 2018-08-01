#include "include/mcm/computation/op/def/maxpool2d.hpp"

mv::op::MaxPool2D::MaxPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
ComputationOp(OpType::MaxPool2D, name),
Pool2DOp(OpType::MaxPool2D, kernelSize, stride, padding, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::MaxPool2D::MaxPool2D(mv::json::Value& obj) :
ComputationOp(obj),
Pool2DOp(obj)
{

}
