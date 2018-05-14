#include "include/fathom/computation/op/kernel_op.hpp"

mv::KernelOp::KernelOp(const Logger &logger, const string &opType, const UnpopulatedTensor &input, Shape kernelShape, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
ComputationOp(logger, opType, input.getDType(), input.getOrder(), input.getShape(), 
Shape(input.getShape()[0], getOutputDim(input.getShape()[1], kernelShape[0], padX, strideX), getOutputDim(input.getShape()[2], kernelShape[1], padY, strideY), kernelShape[2]),
name)
{
    addAttr("strideX", AttrType::ByteType, strideX);
    addAttr("strideY", AttrType::ByteType, strideY);
    addAttr("padX", AttrType::ByteType, padX);
    addAttr("padY", AttrType::ByteType, padY);
}

mv::KernelOp::~KernelOp()
{

}

mv::string mv::KernelOp::toString() const
{
    return ComputationOp::toString();
}
