#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(ops, add)
{
    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> input1Data = mv::utils::generateSequence<double>(256u * 512u);
    std::vector<double> input2Data = mv::utils::generateSequence<double>(256u * 512u);
    auto input1 = om.constant(input1Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    auto input2 = om.constant(input2Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));

    auto add = om.add({input1, input2});
    auto addOp = om.getSourceOp(add);
    auto output = om.output(add);

    ASSERT_EQ(add->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(addOp->getOpType(), "Add");
    ASSERT_EQ(add->attrsCount(), 6);
    ASSERT_EQ(addOp->attrsCount(), 3);
    ASSERT_EQ(addOp->inputSlots(), 2);
    ASSERT_EQ(addOp->outputSlots(), 1);
    ASSERT_TRUE(addOp->hasTypeTrait("executable"));
}
