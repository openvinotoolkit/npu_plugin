#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, concat)
{
    mv::OpModel om("testModel");

    auto input = om.input({8, 8, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));

    std::vector<double> input1Data = mv::utils::generateSequence<double>(input->getShape().totalSize());
    std::vector<double> input2Data = mv::utils::generateSequence<double>(input->getShape().totalSize());
    std::vector<double> input3Data = mv::utils::generateSequence<double>(input->getShape().totalSize());

    auto input1 = om.constant(input1Data, {8, 8, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto input2 = om.constant(input2Data, {8, 8, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto input3 = om.constant(input3Data, {8, 8, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto concat = om.concat({input1, input2, input3});
    auto output = om.output(concat);

    auto concatOp = om.getSourceOp(concat);

    ASSERT_EQ(concat->getShape(), mv::Shape({8, 8, 16*3, 1}));
    ASSERT_EQ(concat->attrsCount(), 6);

    ASSERT_EQ(concatOp->getOpType(), "Concat");
    ASSERT_EQ(concatOp->inputSlots(), 3);
    ASSERT_EQ(concatOp->outputSlots(), 1);
    ASSERT_EQ(concatOp->attrsCount(), 4);
    ASSERT_TRUE(concatOp->hasAttr("axis"));
    ASSERT_EQ(concatOp->get("axis").get<std::string>(), "C");
    ASSERT_TRUE(concatOp->hasTypeTrait("executable"));
}
