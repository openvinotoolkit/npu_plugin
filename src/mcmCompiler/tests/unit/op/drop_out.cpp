#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"


TEST(ops, drop_out)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> inputData = mv::utils::generateSequence<double>(32u * 32u);
    auto input1 = om.constant(inputData, {32, 32}, mv::DType("Float16"), mv::Order("HW"));

    auto dropout = om.dropout(input1);
    auto dropoutOp = om.getSourceOp(dropout);
    auto output = om.output(dropout);

    ASSERT_EQ(dropout->getShape(), mv::Shape({32, 32}));
    ASSERT_EQ(dropoutOp->getOpType(), "Dropout");
    ASSERT_EQ(dropout->attrsCount(), 6);
    ASSERT_EQ(dropoutOp->attrsCount(), 3);
    ASSERT_EQ(dropoutOp->inputSlots(), 1);
    ASSERT_EQ(dropoutOp->outputSlots(), 1);
    ASSERT_TRUE(dropoutOp->hasTypeTrait("exposed"));
}
