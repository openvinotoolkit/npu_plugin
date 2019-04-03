#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, concat)
{

    mv::OpModel om("testModel");

    auto input = om.input({8, 8, 16}, mv::DType("Float16"), mv::Order("CHW"));

    std::vector<double> input1Data = mv::utils::generateSequence<double>(input->getShape().totalSize());
    std::vector<double> input2Data = mv::utils::generateSequence<double>(input->getShape().totalSize());

    auto input1 = om.constant(input1Data, {8, 8, 16}, mv::DType("Float16"), mv::Order("CHW"));
    auto input2 = om.constant(input2Data, {8, 8, 16}, mv::DType("Float16"), mv::Order("CHW"));
    auto concat = om.concat(input1, input2);
    auto output = om.output(concat);

    auto concatOp = om.getSourceOp(concat);

    ASSERT_EQ(concat->getShape(), mv::Shape({8, 8, 32}));
    ASSERT_EQ(concat->attrsCount(), 6);

    ASSERT_EQ(concatOp->getOpType(), "Concat");
    ASSERT_EQ(concatOp->inputSlots(), 2);
    ASSERT_EQ(concatOp->outputSlots(), 1);
    ASSERT_EQ(concatOp->attrsCount(), 3);
    ASSERT_TRUE(concatOp->hasAttr("axis"));
    ASSERT_EQ(concatOp->get("axis").get<std::string>(), "C");
    ASSERT_TRUE(concatOp->hasTypeTrait("executable"));
}
