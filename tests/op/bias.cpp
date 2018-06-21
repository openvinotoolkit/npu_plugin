#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, bias)
{
    mv::OpModel om;
    auto input = om.input(mv::Shape(32, 32, 16), mv::DType::Float, mv::Order::NWHC);
    mv::dynamic_vector<mv::float_type> data = mv::utils::generateSequence<mv::float_type>(16);
    auto biases = om.constant(data, mv::Shape(16), mv::DType::Float, mv::Order::NWHC);
    auto bias = om.bias(input, biases);
    auto biasOp = om.getSourceOp(bias);
    auto output = om.output(bias);

    ASSERT_EQ(output->getAttr("shape").getContent<mv::Shape>(), mv::Shape(32, 32, 16));
    ASSERT_EQ(biasOp->getOpType(), mv::OpType::Bias);
    ASSERT_EQ(biasOp->attrsCount(), 8);
    ASSERT_EQ(biasOp->inputSlots(), 2);
    ASSERT_EQ(biasOp->outputSlots(), 1);
    ASSERT_TRUE(biasOp->isExecutable());

}