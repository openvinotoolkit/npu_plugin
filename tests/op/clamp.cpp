#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(ops, clamp)
{
    constexpr int N=1, C=3, H=200, W=320;
    constexpr double min=0, max=6;

    mv::OpModel om("testModel");

    std::vector<double> inputData = mv::utils::generateSequence<double>(N * C * H * W);
    auto input = om.constant(inputData, {W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto clamp = om.clamp(input, min, max);
    auto output = om.output(clamp);

    ASSERT_EQ(clamp->getShape(), mv::Shape({W, H, C, N}));
    ASSERT_EQ(clamp->attrsCount(), 6);

    auto clampOp = om.getSourceOp(clamp);
    ASSERT_EQ(clampOp->getOpType(), "Clamp");
    ASSERT_EQ(clampOp->attrsCount(), 4);
    ASSERT_EQ(clampOp->inputSlots(), 1);
    ASSERT_EQ(clampOp->outputSlots(), 1);
    ASSERT_TRUE(clampOp->hasTypeTrait("executable"));
}
