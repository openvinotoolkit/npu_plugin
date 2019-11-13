#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, bias)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> data = mv::utils::generateSequence<double>(16);
    auto biases = om.constant(data, {16}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)));
    auto bias = om.bias(input, biases);
    auto biasOp = om.getSourceOp(bias);
    om.output(bias);

    ASSERT_EQ(bias->getShape(), mv::Shape({32, 32, 16, 1}));
    ASSERT_EQ(biasOp->getOpType(), "Bias");
    ASSERT_EQ(biasOp->attrsCount(), 3);
    ASSERT_EQ(biasOp->inputSlots(), 2);
    ASSERT_EQ(biasOp->outputSlots(), 1);
    ASSERT_TRUE(biasOp->hasTypeTrait("executable"));

}
