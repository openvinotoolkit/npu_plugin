#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, matMul)
{
    mv::OpModel om("testModel");
    auto input0 = om.input({256, 512}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));
    std::vector<double> input1Data = mv::utils::generateSequence<double>(256u * 512u);
    std::vector<double> input2Data = mv::utils::generateSequence<double>(256u * 512u);
    auto input1 = om.constant(input1Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    auto input2 = om.constant(input2Data, {512, 256}, mv::DType("Float16"), mv::Order("HW"));

    auto matmul = om.matMul(input1, input2);
    auto matmulOp = om.getSourceOp(matmul);
    auto output = om.output(matmul);

    ASSERT_EQ(matmul->getShape(), mv::Shape({256, 256}));
    ASSERT_EQ(matmulOp->getOpType(), "MatMul");
    ASSERT_EQ(matmul->attrsCount(), 6);
    ASSERT_EQ(matmulOp->attrsCount(), 3);
    ASSERT_EQ(matmulOp->inputSlots(), 2);
    ASSERT_EQ(matmulOp->outputSlots(), 1);
    ASSERT_TRUE(matmulOp->hasTypeTrait("executable"));

}
