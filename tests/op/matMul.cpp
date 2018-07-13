#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, matMul)
{

    mv::OpModel om;
    auto input0 = om.input(mv::Shape(256, 512), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(512u * 100u);
    auto input1 = om.constant(weightsData, mv::Shape(512, 100), mv::DType::Float, mv::Order::ColumnMajor);
    auto matMul = om.matMul(input0, input1);
    auto matMulOp = om.getSourceOp(matMul);
    auto output = om.output(matMul);

    ASSERT_EQ(output->getShape(), mv::Shape(256, 100));
    ASSERT_EQ(matMulOp->getOpType(), mv::OpType::MatMul);
    ASSERT_EQ(matMulOp->attrsCount(), 8);
    ASSERT_EQ(matMulOp->inputSlots(), 2);
    ASSERT_EQ(matMulOp->outputSlots(), 1);
    ASSERT_TRUE(matMulOp->isExecutable());

}
