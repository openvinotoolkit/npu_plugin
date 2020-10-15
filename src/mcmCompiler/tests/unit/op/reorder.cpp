#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, reorder)
{

    // Note, that the Reorder operation does not touch tensor data,
    // but only formally reassigns new order attribute to the tensor.
    //
    // For example, if input tensor is ordered as "NCHW", assigning
    // new order like "NCWH" to the output tensor essentially means,
    // that now we consider output tensor's fastest index as height
    // instead of width.
    //
    // The effect of such reordering of dimensions is same as if we
    // transposed the image, without physically transposing of data.

    constexpr int N = 8;   // batch
    constexpr int C = 3;   // channels
    constexpr int H = 200; // height
    constexpr int W = 320; // width

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto reorder = om.reorder(input, mv::Order("NCWH")); // rename W and H dims
    auto output = om.output(reorder);

    auto reorderOp = om.getSourceOp(reorder);

    ASSERT_EQ(reorder->getOrder(), mv::Order("NCWH"));       // H and W permuted
    ASSERT_EQ(reorder->getShape(), mv::Shape({W, H, C, N})); // same as input tensor
    ASSERT_EQ(reorder->attrsCount(), 6);

    ASSERT_EQ(reorderOp->getOpType(), "Reorder");
    ASSERT_EQ(reorderOp->attrsCount(), 3);
    ASSERT_EQ(reorderOp->inputSlots(), 1);
    ASSERT_EQ(reorderOp->outputSlots(), 1);
    ASSERT_TRUE(reorderOp->hasTypeTrait("executable"));

}
