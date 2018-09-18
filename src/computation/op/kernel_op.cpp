#include "include/mcm/computation/op/kernel_op.hpp"

mv::KernelOp::KernelOp(OpType opType, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name) :
ComputationOp(opType, name),
SourceOp(opType, 1, name)
{
    set<std::array<short unsigned, 2>>("stride", stride);
    set<std::array<short unsigned, 4>>("padding", padding);

}

mv::KernelOp::~KernelOp()
{

}
