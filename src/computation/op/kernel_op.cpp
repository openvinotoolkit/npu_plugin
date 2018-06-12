#include "include/fathom/computation/op/kernel_op.hpp"

mv::KernelOp::KernelOp(OpType opType, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
ComputationOp(opType, name),
SourceOp(opType, 1, name)
{
    addAttr("stride", AttrType::UnsignedVec2DType, stride);
    addAttr("padding", AttrType::UnsignedVec4DType, padding);

}

mv::KernelOp::~KernelOp()
{

}