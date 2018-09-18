#include "include/mcm/computation/op/def/maxpool2d.hpp"

mv::op::MaxPool2D::MaxPool2D(std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name) :
ComputationOp(OpType::MaxPool2D, name),
Pool2DOp(OpType::MaxPool2D, kernelSize, stride, padding, name)
{
    set<bool>("executable", true);
}

bool mv::op::MaxPool2D::isHardwarizeable(json::Object&)
{
    return false;
}
