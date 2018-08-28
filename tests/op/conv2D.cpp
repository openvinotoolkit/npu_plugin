#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, conv2D)
{

    mv::OpModel om;
    auto input = om.input({32, 32, 1}, mv::DType::Float, mv::Order::ColumnMajor);
    std::vector<double> weightsData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f});
    auto weights1 = om.constant(weightsData, {3, 3, 1, 3}, mv::DType::Float, mv::Order::ColumnMajor);
    auto conv = om.conv2D(input, weights1, {4, 4}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    auto output = om.output(conv);

    ASSERT_EQ(output->getShape(), mv::Shape({8, 8, 3}));
    ASSERT_EQ(convOp->getOpType(), mv::OpType::Conv2D);
    ASSERT_EQ(convOp->attrsCount(), 9);
    ASSERT_EQ(convOp->getAttr("stride").getType(), mv::AttrType::UnsignedVec2DType);
    ASSERT_EQ(convOp->getAttr("padding").getType(), mv::AttrType::UnsignedVec4DType);
    ASSERT_EQ(convOp->getAttr("stride").getContent<mv::UnsignedVector2D>().e0, 4);
    ASSERT_EQ(convOp->getAttr("stride").getContent<mv::UnsignedVector2D>().e1, 4);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e0, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e1, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e2, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e3, 1);
    ASSERT_EQ(convOp->inputSlots(), 2);
    ASSERT_EQ(convOp->outputSlots(), 1);
    ASSERT_TRUE(convOp->isExecutable());

}
