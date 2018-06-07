#include "include/fathom/computation/op/kernel_op.hpp"

mv::KernelOp::KernelOp(const Logger &logger, const string &opType, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
ComputationOp(logger, opType, name),
SourceOp(logger, opType, name)
{
    addAttr("executable", AttrType::BoolType, true);
    addAttr("strideX", AttrType::ByteType, strideX);
    addAttr("strideY", AttrType::ByteType, strideY);
    addAttr("padX", AttrType::ByteType, padX);
    addAttr("padY", AttrType::ByteType, padY);
}

mv::KernelOp::~KernelOp()
{

}