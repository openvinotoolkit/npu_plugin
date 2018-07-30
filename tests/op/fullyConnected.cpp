#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, fullyConnected)
{

    mv::OpModel om;
    auto input = om.input(mv::Shape(8, 8, 16), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(input->getShape().totalSize() * 100u);
    auto weights1 = om.constant(weightsData, mv::Shape(input->getShape().totalSize(), 100), mv::DType::Float, mv::Order::ColumnMajor);
    auto fullyConnected = om.fullyConnected(input, weights1);
    auto fullyConnectedOp = om.getSourceOp(fullyConnected);
    auto output = om.output(fullyConnected);

    ASSERT_EQ(output->getShape(), mv::Shape(1, 100));
    ASSERT_EQ(fullyConnectedOp->getOpType(), mv::OpType::FullyConnected);
    ASSERT_EQ(fullyConnectedOp->attrsCount(), 7);
    ASSERT_EQ(fullyConnectedOp->inputSlots(), 2);
    ASSERT_EQ(fullyConnectedOp->outputSlots(), 1);
    ASSERT_TRUE(fullyConnectedOp->isExecutable());

}
