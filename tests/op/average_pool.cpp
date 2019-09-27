#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, averagePool2D)
{

    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});
    auto weights1 = om.constant(weightsData, {3, 3, 1, 3}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv = om.conv(input, weights1, {4, 4}, {1, 1, 1, 1}, 1);
    auto pool = om.averagePool(conv, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto poolOp = om.getSourceOp(pool);
    auto output = om.output(pool);


    ASSERT_EQ(pool->getShape(), mv::Shape({4, 4, 3, 1}));
    ASSERT_EQ(poolOp->getOpType(), "AveragePool");
    //ASSERT_EQ(convOp->attrsCount(), 9);
    auto s0 = poolOp->get<std::array<unsigned short, 2>>("stride")[0];
    auto s1 = poolOp->get<std::array<unsigned short, 2>>("stride")[1];
    ASSERT_EQ(s0, 2);
    ASSERT_EQ(s1, 2);
    auto p0 = poolOp->get<std::array<unsigned short, 4>>("padding")[0];
    auto p1 = poolOp->get<std::array<unsigned short, 4>>("padding")[1];
    auto p2 = poolOp->get<std::array<unsigned short, 4>>("padding")[2];
    auto p3 = poolOp->get<std::array<unsigned short, 4>>("padding")[3];
    ASSERT_EQ(p0, 1);
    ASSERT_EQ(p1, 1);
    ASSERT_EQ(p2, 1);
    ASSERT_EQ(p3, 1);
    ASSERT_EQ(poolOp->inputSlots(), 1);
    ASSERT_EQ(poolOp->outputSlots(), 1);
    ASSERT_EQ(poolOp->get<bool>("exclude_pad"), true);
    ASSERT_EQ(poolOp->get<std::string>("auto_pad"), "");
    ASSERT_EQ(poolOp->get<std::string>("rounding_type"), "floor");
    ASSERT_TRUE(poolOp->hasTypeTrait("executable"));

}
