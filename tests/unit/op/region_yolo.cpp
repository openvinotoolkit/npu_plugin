#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, region_yolo_num)
{
    constexpr int N = 1;   // batch
    constexpr int C = 125; // channels
    constexpr int H = 13;  // height
    constexpr int W = 13;  // width

    unsigned coords = 4;
    unsigned classes = 20;
    bool do_softmax = false;
    unsigned num = 5;

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto region = om.regionYolo(input, coords, classes, do_softmax, num);
    auto output = om.output(region);

    auto regionOp = om.getSourceOp(region);

    ASSERT_EQ(region->getOrder(), mv::Order("NC"));
    ASSERT_EQ(region->getShape(), mv::Shape({H * W * C, N}));
    ASSERT_EQ(region->attrsCount(), 6);

    ASSERT_EQ(regionOp->getOpType(), "RegionYolo");
    ASSERT_EQ(regionOp->attrsCount(), 7);
    ASSERT_EQ(regionOp->inputSlots(), 1);
    ASSERT_EQ(regionOp->outputSlots(), 1);
    ASSERT_TRUE(regionOp->hasTypeTrait("executable"));
}

TEST(ops, region_yolo_mask)
{
    constexpr int N = 1;  // batch
    constexpr int C = 75; // channels
    constexpr int H = 13; // height
    constexpr int W = 13; // width

    unsigned coords = 4;
    unsigned classes = 20;
    bool do_softmax = true;
    std::vector<unsigned> mask({0, 1, 2});

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto region = om.regionYolo(input, coords, classes, do_softmax, 0, mask);
    auto output = om.output(region);

    auto regionOp = om.getSourceOp(region);

    ASSERT_EQ(region->getOrder(), mv::Order("NC"));
    ASSERT_EQ(region->getShape(), mv::Shape({H * W * C, N}));
    ASSERT_EQ(region->attrsCount(), 6);

    ASSERT_EQ(regionOp->getOpType(), "RegionYolo");
    ASSERT_EQ(regionOp->attrsCount(), 7);
    ASSERT_EQ(regionOp->inputSlots(), 1);
    ASSERT_EQ(regionOp->outputSlots(), 1);
    ASSERT_TRUE(regionOp->hasTypeTrait("executable"));
}
