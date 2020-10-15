#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(ops, batchnorm)
{
    mv::OpModel om("testModel");
    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> bnInputData = mv::utils::generateSequence<double>(input->getShape()[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> meanData = mv::utils::generateSequence<double>(input->getShape()[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> varData = mv::utils::generateSequence<double>(input->getShape()[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(input->getShape()[mv::IO_CHANNEL_DIMENSION]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(input->getShape()[mv::IO_CHANNEL_DIMENSION]);

    auto bnmean = om.constant(meanData, {input->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"));
    auto bnvariance = om.constant(varData, {input->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"));
    auto bnoffset = om.constant(offsetData, {input->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"));
    auto bnscale = om.constant(scaleData, {input->getShape()[mv::IO_CHANNEL_DIMENSION]}, mv::DType("Float16"), mv::Order("W"));
    double eps = 1e-9;

    auto batchnorm = om.batchNormalization(input, bnmean, bnvariance, bnoffset, bnscale, eps);
    auto bnOp = om.getSourceOp(batchnorm);
    auto output = om.output(batchnorm);

    ASSERT_EQ(batchnorm->getShape(), mv::Shape({224, 224, 3, 1}));
    ASSERT_EQ(bnOp->getOpType(), "BatchNormalization");
    ASSERT_EQ(batchnorm->attrsCount(), 6);
    ASSERT_EQ(bnOp->attrsCount(), 4);
    ASSERT_EQ(bnOp->inputSlots(), 5);
    ASSERT_EQ(bnOp->outputSlots(), 1);
    ASSERT_TRUE(bnOp->hasTypeTrait("executable"));
}

