#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, reorg_yolo)
{
    // Reorg Yolo reorganizes tensor like,
    // e.g. NxCxHxW into Nx(C*4)x(H/2)x(W/2)

    constexpr int N = 8;   // batch
    constexpr int C = 3;   // channels
    constexpr int H = 200; // height
    constexpr int W = 320; // width

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto reorg = om.reorgYolo(input, 2);
    auto output = om.output(reorg);

    auto reorgOp = om.getSourceOp(reorg);

    ASSERT_EQ(reorg->getOrder(), mv::Order("NCHW")); // same as input tensor
    ASSERT_EQ(reorg->getShape(), mv::Shape({W/2, H/2, C*4, N}));
    ASSERT_EQ(reorg->attrsCount(), 6);

    ASSERT_EQ(reorgOp->getOpType(), "ReorgYolo");
    ASSERT_EQ(reorgOp->attrsCount(), 3);
    ASSERT_EQ(reorgOp->inputSlots(), 1);
    ASSERT_EQ(reorgOp->outputSlots(), 1);
    ASSERT_TRUE(reorgOp->hasTypeTrait("executable"));
}
