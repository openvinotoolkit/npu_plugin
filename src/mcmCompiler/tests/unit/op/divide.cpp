#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, divide)
{

    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> input1Data = mv::utils::generateSequence<double>(256u * 512u);
    std::vector<double> input2Data = mv::utils::generateSequence<double>(256u * 512u);
    auto input1 = om.constant(input1Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    auto input2 = om.constant(input2Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));

    auto divide = om.divide(input1, input2);
    auto divideOp = om.getSourceOp(divide);
    auto output = om.output(divide);

    ASSERT_EQ(divide->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(divideOp->getOpType(), "Divide");
    ASSERT_EQ(divide->attrsCount(), 6);
    ASSERT_EQ(divideOp->attrsCount(), 2);
    ASSERT_EQ(divideOp->inputSlots(), 2);
    ASSERT_EQ(divideOp->outputSlots(), 1);
    ASSERT_TRUE(divideOp->hasTypeTrait("executable"));

}
