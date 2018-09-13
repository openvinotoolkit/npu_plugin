#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, matMul)
{

    mv::OpModel om;
    auto input0 = om.input({256, 512}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> weightsData = mv::utils::generateSequence<double>(512u * 100u);
    auto input1 = om.constant(weightsData, {512, 100}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto matMul = om.matMul(input0, input1);
    auto matMulOp = om.getSourceOp(matMul);
    auto output = om.output(matMul);

    ASSERT_EQ(output->getShape(), mv::Shape({256, 100}));
    ASSERT_EQ(matMulOp->getOpType(), mv::OpType::MatMul);
    ASSERT_EQ(matMulOp->attrsCount(), 7);
    ASSERT_EQ(matMulOp->inputSlots(), 2);
    ASSERT_EQ(matMulOp->outputSlots(), 1);
    ASSERT_TRUE(matMulOp->isExecutable());

}
