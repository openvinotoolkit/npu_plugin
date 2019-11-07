#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, multiply)
{

    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> input1Data = mv::utils::generateSequence<double>(256u * 512u);
    std::vector<double> input2Data = mv::utils::generateSequence<double>(256u * 512u);
    auto input1 = om.constant(input1Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    auto input2 = om.constant(input2Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));

    auto multiply = om.multiply({input1, input2});
    auto multiplyOp = om.getSourceOp(multiply);
    auto output = om.output(multiply);

    ASSERT_EQ(multiply->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(multiplyOp->getOpType(), "Multiply");
    ASSERT_EQ(multiply->attrsCount(), 6);
    ASSERT_EQ(multiplyOp->attrsCount(), 3);
    ASSERT_EQ(multiplyOp->inputSlots(), 2);
    ASSERT_EQ(multiplyOp->outputSlots(), 1);
    ASSERT_TRUE(multiplyOp->hasTypeTrait("executable"));

}
