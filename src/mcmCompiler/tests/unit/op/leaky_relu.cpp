#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, leaky_relu_default)
{

    mv::OpModel om("testModel");
    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"),  mv::Order("NHWC"));

    // Omit default parameter alpha when instantiating the LeakyRelu object.
    auto leakyRelu = om.leakyRelu(input);
    auto leakyReluOp = om.getSourceOp(leakyRelu);
    auto output = om.output(leakyRelu);
    auto alpha = leakyReluOp->get<double>("alpha");
    
    ASSERT_EQ(leakyReluOp->getOpType(), "LeakyRelu");
    ASSERT_EQ(alpha, 0.0);
    
}

TEST(ops, leaky_relu_custom)
{

    mv::OpModel om("testModel");
    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"),  mv::Order("NHWC"));

    auto leakyRelu = om.leakyRelu(input, 0.1);
    auto leakyReluOp = om.getSourceOp(leakyRelu);
    auto output = om.output(leakyRelu);
    auto alpha = leakyReluOp->get<double>("alpha");
    
    ASSERT_EQ(leakyReluOp->getOpType(), "LeakyRelu");
    ASSERT_EQ(alpha, 0.1);

}
