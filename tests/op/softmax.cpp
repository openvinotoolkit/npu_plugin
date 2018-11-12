#include "gtest/gtest.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, softmax)
{
    mv::OpModel om("testModel");
    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto softmax = om.softmax(input);
    auto softmaxOp = om.getSourceOp(softmax);
    auto output = om.output(softmax);

    ASSERT_EQ(softmax->getShape(), mv::Shape({32, 32, 3}));
    ASSERT_EQ(softmaxOp->getOpType(), "Softmax");
    ASSERT_EQ(softmaxOp->attrsCount(), 2);
    ASSERT_EQ(softmaxOp->inputSlots(), 1);
    ASSERT_EQ(softmaxOp->outputSlots(), 1);
    ASSERT_EQ(softmax->attrsCount(), 6);
    ASSERT_TRUE(softmaxOp->hasTypeTrait("executable"));

}
