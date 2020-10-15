#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, fullyConnected)
{
    mv::OpModel om("testModel");
    auto input = om.input({8, 8, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));

    std::vector<double> weightsData = mv::utils::generateSequence<double>(input->getShape().totalSize() * 100u);
    auto weights1 = om.constant(weightsData, {input->getShape().totalSize(), 100}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(2)));
    auto fullyConnected = om.fullyConnected(input, weights1);
    auto fullyConnectedOp = om.getSourceOp(fullyConnected);
    auto output = om.output(fullyConnected);

    ASSERT_EQ(fullyConnected->getShape(), mv::Shape({1, 100}));
    ASSERT_EQ(fullyConnectedOp->getOpType(), "FullyConnected");
    ASSERT_EQ(fullyConnectedOp->inputSlots(), 2);
    ASSERT_EQ(fullyConnectedOp->outputSlots(), 1);
    ASSERT_EQ(fullyConnectedOp->attrsCount(), 3);
    ASSERT_EQ(fullyConnected->attrsCount(), 6);
    ASSERT_TRUE(fullyConnectedOp->hasTypeTrait("executable"));
}
