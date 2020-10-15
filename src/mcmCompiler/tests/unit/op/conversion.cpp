#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, conversion)
{

    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> data = mv::utils::generateSequence<double>(256u * 512u);
    auto dataTensor = om.constant(data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));

    auto conversion = om.conversion(dataTensor, mv::Order("WH"));
    auto conversionOp = om.getSourceOp(conversion);
    auto output = om.output(conversion);

    ASSERT_EQ(conversion->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(conversion->getOrder(), mv::Order("WH"));
    ASSERT_EQ(conversionOp->getOpType(), "Conversion");
    ASSERT_EQ(conversion->attrsCount(), 6);
    ASSERT_EQ(conversionOp->attrsCount(), 3);
    ASSERT_EQ(conversionOp->inputSlots(), 1);
    ASSERT_EQ(conversionOp->outputSlots(), 1);
    ASSERT_TRUE(conversionOp->hasTypeTrait("executable"));

}
