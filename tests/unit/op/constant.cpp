#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(ops, constant)
{
    mv::OpModel om("TestModel");
    auto input = om.input({32, 32}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> constantData = mv::utils::generateSequence<double>(32u * 32u);
    auto constant = om.constant(constantData, {32, 32}, mv::DType("Float16"), mv::Order("HW"));
    auto constantOp = om.getSourceOp(constant);

    ASSERT_EQ(constant->getShape(), mv::Shape({32, 32}));
    ASSERT_EQ(constantOp->getOpType(), "Constant");
    ASSERT_EQ(constant->attrsCount(), 5);
    ASSERT_EQ(constantOp->attrsCount(), 7);
    ASSERT_EQ(constantOp->inputSlots(), 0);
    ASSERT_EQ(constantOp->outputSlots(), 1);

}
