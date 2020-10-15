#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, scale)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> data = mv::utils::generateSequence<double>(16);
    auto weights = om.constant(data, {16}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)));
    auto layer = om.scale(input, weights);
    auto layerOp = om.getSourceOp(layer);
    om.output(layer);

    ASSERT_EQ(layer->getShape(), mv::Shape({32, 32, 16, 1}));
    ASSERT_EQ(layerOp->getOpType(), "Scale");
    ASSERT_EQ(layerOp->attrsCount(), 3);
    ASSERT_EQ(layerOp->inputSlots(), 2);
    ASSERT_EQ(layerOp->outputSlots(), 1);
    ASSERT_TRUE(layerOp->hasTypeTrait("executable"));
}
