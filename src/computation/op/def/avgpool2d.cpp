#include "include/mcm/computation/op/def/avgpool2d.hpp"

mv::op::AvgPool2D::AvgPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
ComputationOp(OpType::AvgPool2D, name),
Pool2DOp(OpType::AvgPool2D, kernelSize, stride, padding, name)
{
    addAttr("executable", AttrType::BoolType, true);
}