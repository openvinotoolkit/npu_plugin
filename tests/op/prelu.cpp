#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, prelu)
{
    mv::OpModel om;
    auto input = om.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(3);
    auto slope = om.constant(data, mv::Shape(3), mv::DType::Float, mv::Order::ColumnMajor);

    auto prelu = om.prelu(input, slope);
    auto preluOp = om.getSourceOp(prelu);
    auto output = om.output(prelu);

    ASSERT_EQ(output->getShape(), mv::Shape(32, 32, 3));
    ASSERT_EQ(preluOp->getOpType(), mv::OpType::PReLU);
    ASSERT_EQ(preluOp->attrsCount(), 7);
    ASSERT_EQ(preluOp->inputSlots(), 2);
    ASSERT_EQ(preluOp->outputSlots(), 1);
    ASSERT_TRUE(preluOp->isExecutable());

}
