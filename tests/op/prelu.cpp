#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, prelu)
{
    /*mv::OpModel om("testModel");
    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> data = mv::utils::generateSequence<double>(3);
    auto slope = om.constant(data, {3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);

    auto prelu = om.prelu(input, slope);
    auto preluOp = om.getSourceOp(prelu);
    auto output = om.output(prelu);

    ASSERT_EQ(output->getShape(), mv::Shape({32, 32, 3}));
    ASSERT_EQ(preluOp->getOpType(), mv::OpType::PReLU);
    ASSERT_EQ(preluOp->attrsCount(), 7);
    ASSERT_EQ(preluOp->inputSlots(), 2);
    ASSERT_EQ(preluOp->outputSlots(), 1);
    ASSERT_TRUE(preluOp->isExecutable());*/

}
