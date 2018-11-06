#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, reshape)
{
    mv::OpModel om("testModel");
    auto input = om.input({10, 10, 5}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto reshape = om.reshape(input, {5, 5, 20});
    auto reshapeOp = om.getSourceOp(reshape);
    om.output(reshape);

    ASSERT_EQ(reshape->getShape(), mv::Shape({5, 5, 20}));
    ASSERT_EQ(reshapeOp->getOpType(), "Reshape");
    ASSERT_EQ(reshapeOp->attrsCount(), 3);
    ASSERT_EQ(reshape->attrsCount(), 5);
    ASSERT_EQ(reshapeOp->inputSlots(), 1);
    ASSERT_EQ(reshapeOp->outputSlots(), 1);
    //ASSERT_TRUE(biasOp->isExecutable());

}