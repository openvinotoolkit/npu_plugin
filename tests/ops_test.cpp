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
    om.output(bias);

    auto biasOp = om.getSourceOp(bias);

    ASSERT_EQ(bias->getShape(), mv::Shape(32, 32, 16));
    ASSERT_EQ(biasOp->getOpType(), mv::OpType::Bias);

}