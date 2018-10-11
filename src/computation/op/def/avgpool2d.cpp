#include "include/mcm/computation/op/def/avgpool2d.hpp"

mv::op::AvgPool2D::AvgPool2D(std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name) :
ComputationOp(OpType::AvgPool2D, name),
Pool2DOp(OpType::AvgPool2D, kernelSize, stride, padding, name)
{
    set<bool>("executable", true);
}

bool mv::op::AvgPool2D::isHardwarizeable(json::Object&)
{
    return true;
}
