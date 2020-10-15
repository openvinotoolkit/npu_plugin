#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, reshape)
{
    mv::OpModel om("testModel");
    auto input = om.input({10, 10, 5, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto reshape = om.reshape(input, {5, 5, 20, 1});
    auto reshapeOp = om.getSourceOp(reshape);
    om.output(reshape);

    ASSERT_EQ(reshape->getShape(), mv::Shape({5, 5, 20, 1}));
    ASSERT_EQ(reshape->getOrder(), mv::Order("NCHW"));
    ASSERT_EQ(reshape->attrsCount(), 6);

    ASSERT_EQ(reshapeOp->getOpType(), "Reshape");
    ASSERT_EQ(reshapeOp->attrsCount(), 4);
    ASSERT_EQ(reshapeOp->inputSlots(), 1);
    ASSERT_EQ(reshapeOp->outputSlots(), 1);
    ASSERT_TRUE(reshapeOp->hasTypeTrait("executable"));
}

TEST(ops, reshape_reorder)
{
    const unsigned N=8, C=3, H=200, W=320;

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto reshape = om.reshape(input, {W*H*C, N}, "NC"); // reshape, and new order
    auto reshapeOp = om.getSourceOp(reshape);
    om.output(reshape);

    ASSERT_EQ(reshape->getShape(), mv::Shape({W*H*C, N}));
    ASSERT_EQ(reshape->getOrder(), mv::Order("NC"));
    ASSERT_EQ(reshape->attrsCount(), 6);

    ASSERT_EQ(reshapeOp->getOpType(), "Reshape");
    ASSERT_EQ(reshapeOp->attrsCount(), 4);
    ASSERT_EQ(reshapeOp->inputSlots(), 1);
    ASSERT_EQ(reshapeOp->outputSlots(), 1);
    ASSERT_TRUE(reshapeOp->hasTypeTrait("executable"));
}
