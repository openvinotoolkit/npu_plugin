#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, bias)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 16}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    std::vector<double> data = mv::utils::generateSequence<double>(16);
    auto biases = om.constant(data, {16}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(1)));
    auto bias = om.bias(input, biases);
    auto biasOp = om.getSourceOp(bias);
    auto output = om.output(bias);

    ASSERT_EQ(output->getShape(), mv::Shape({32, 32, 16}));
    ASSERT_EQ(biasOp->getOpType(), mv::OpType::Bias);
    ASSERT_EQ(biasOp->attrsCount(), 7);
    ASSERT_EQ(biasOp->inputSlots(), 2);
    ASSERT_EQ(biasOp->outputSlots(), 1);
    ASSERT_TRUE(biasOp->isExecutable());

}
