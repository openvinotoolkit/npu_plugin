#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, prelu)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> data = mv::utils::generateSequence<double>(3);
    auto slope = om.constant(data, {3}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)));

    auto prelu = om.prelu(input, slope);
    auto preluOp = om.getSourceOp(prelu);
    auto output = om.output(prelu);

    ASSERT_EQ(prelu->getShape(), mv::Shape({32, 32, 3, 1}));
    ASSERT_EQ(preluOp->getOpType(), "Prelu");
    ASSERT_EQ(preluOp->attrsCount(), 2);
    ASSERT_EQ(preluOp->inputSlots(), 2);
    ASSERT_EQ(preluOp->outputSlots(), 1);
    ASSERT_EQ(prelu->attrsCount(), 6);
    ASSERT_TRUE(preluOp->hasTypeTrait("executable"));

}
