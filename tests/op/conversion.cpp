#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, conversion)
{

    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> data = mv::utils::generateSequence<double>(256u * 512u);
    auto dataTensor = om.constant(data, {256, 512}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);

    auto conversion = om.conversion(dataTensor, mv::OrderType::RowMajor);
    auto conversionOp = om.getSourceOp(conversion);
    auto output = om.output(conversion);

    ASSERT_EQ(conversion->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(conversion->getOrder(), mv::OrderType::RowMajor);
    ASSERT_EQ(conversionOp->getOpType(), "Conversion");
    ASSERT_EQ(conversion->attrsCount(), 5);
    ASSERT_EQ(conversionOp->attrsCount(), 3);
    ASSERT_EQ(conversionOp->inputSlots(), 1);
    ASSERT_EQ(conversionOp->outputSlots(), 1);
    //ASSERT_TRUE(subtractOp->isExecutable());

}
