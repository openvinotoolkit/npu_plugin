#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, depthconv2D)
{

    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 2, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                     8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f});
    auto weights1 = om.constant(weightsData, {3, 3, 2, 1}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    auto conv = om.depthwiseConv(input, weights1, {4, 4}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto output = om.output(conv);


    ASSERT_EQ(conv->getShape(), mv::Shape({8, 8, 2, 1}));
    ASSERT_EQ(convOp->getOpType(), "DepthwiseConv");
    ASSERT_EQ(convOp->attrsCount(), 6);
    auto s0 = convOp->get<std::array<unsigned short, 2>>("stride")[0];
    auto s1 = convOp->get<std::array<unsigned short, 2>>("stride")[1];
    ASSERT_EQ(s0, 4);
    ASSERT_EQ(s1, 4);
    auto p0 = convOp->get<std::array<unsigned short, 4>>("padding")[0];
    auto p1 = convOp->get<std::array<unsigned short, 4>>("padding")[1];
    auto p2 = convOp->get<std::array<unsigned short, 4>>("padding")[2];
    auto p3 = convOp->get<std::array<unsigned short, 4>>("padding")[3];
    ASSERT_EQ(p0, 1);
    ASSERT_EQ(p1, 1);
    ASSERT_EQ(p2, 1);
    ASSERT_EQ(p3, 1);
    ASSERT_EQ(convOp->inputSlots(), 2);
    ASSERT_EQ(convOp->outputSlots(), 1);
    ASSERT_TRUE(convOp->hasTypeTrait("executable"));

}
