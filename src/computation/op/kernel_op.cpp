#include "include/mcm/computation/op/kernel_op.hpp"

mv::KernelOp::KernelOp(OpType opType, UnsignedVector2D stride, UnsignedVector4D padding, const std::string &name) :
ComputationOp(opType, name),
SourceOp(opType, 1, name)
{
    addAttr("stride", AttrType::UnsignedVec2DType, stride);
    addAttr("padding", AttrType::UnsignedVec4DType, padding);

}

mv::KernelOp::KernelOp(mv::json::Value& value) :
ComputationOp(value),
SourceOp(value)
{

}

mv::KernelOp::~KernelOp()
{

}
