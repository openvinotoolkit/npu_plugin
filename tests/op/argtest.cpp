/**
 * @brief Example presenting generation of Caffe prototxt and CaffeModel files
 * 
 * In this example a model is composed using MCMCompiler's Composition API. Then
 * the compilation is for target MA2480 is initialized and compilation passes scheduled by 
 * target descriptor are executed. Included GenerateCaffe pass will generate Caffe files.
 * 
 */

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, argtest)
{

    mv::OpModel om("argTestModel");
    auto input = om.input({224, 224, 3}, mv::DTypeType::Float16,  mv::Order("HWC"));

    // Omit default parameter alpha when instantiating the LeakyRelu object.
    auto leakyRelu = om.leakyRelu(input);
    auto leakyReluOp = om.getSourceOp(leakyRelu);
    auto output = om.output(leakyRelu);
    auto alpha = leakyReluOp->get<unsigned>("alpha");
    
    ASSERT_EQ(leakyReluOp->getOpType(), "LeakyRelu");

    // Test default parameter
    ASSERT_EQ(alpha, 1);
}
