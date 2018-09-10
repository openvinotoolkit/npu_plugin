#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, prelu)
{
    mv::OpModel om;
    auto input = om.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(3);
    auto slope = om.constant(data, mv::Shape(3), mv::DType::Float, mv::Order::ColumnMajor);

    std::cout << "CREATE THE PRELU" << std::endl;
    auto prelu = om.prelu(input, slope);
    std::cout << "GET tHE sOURCEOP" << std::endl;
    auto preluOp = om.getSourceOp(prelu);
    std::cout << "Add output" << std::endl;
    auto output = om.output(prelu);

    std::cout << "Output shape:" << mv::Printable::toString(output->getShape()) << std::endl;

    ASSERT_EQ(output->getShape(), mv::Shape(32, 32, 3));
    ASSERT_EQ(preluOp->getOpType(), mv::OpType::PReLU);
    ASSERT_EQ(preluOp->attrsCount(), 7);
    ASSERT_EQ(preluOp->inputSlots(), 2);
    ASSERT_EQ(preluOp->outputSlots(), 1);
    ASSERT_TRUE(preluOp->isExecutable());

}