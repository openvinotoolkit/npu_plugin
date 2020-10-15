#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, permute)
{

    // Note, that the Permute operation does not change the order of the tensor!
    // That is: the output tensor would have the same order as the input one.
    //
    // Instead, the Permute would physically transpose the data in the tensor;
    // and the "order" parameter defines -- which of dimensions to transpose.
    // Namely, the "old" order of the input tensor and the "new" order given
    // by the Permute order parameter would define the dimensions permutation.
    //
    // For example, if input tensor's order is "NCHW" and Permute's parameter
    // order is "NCWH", this means permuting the dimensions "W" and "H".
    // That is: if input tensor shape were 8x3x200x320, the output tensor
    // shape would become 8x3x320x200, while order still remain "NCHW".
    // This means: transposing image H0=200, W0=320 into H1=320, W1=200.

    constexpr int N = 8;   // batch
    constexpr int C = 3;   // channels
    constexpr int H = 200; // height
    constexpr int W = 320; // width

    mv::OpModel om("testModel");
    auto input = om.input({W, H, C, N}, mv::DType("Float16"), mv::Order("NCHW"));
    auto permute = om.permute(input, mv::Order("NCWH")); // transpose W and H
    auto output = om.output(permute);

    auto permuteOp = om.getSourceOp(permute);

    ASSERT_EQ(permute->getOrder(), mv::Order("NCHW"));       // same as input tensor
    ASSERT_EQ(permute->getShape(), mv::Shape({H, W, C, N})); // H and W permuted
    ASSERT_EQ(permute->attrsCount(), 6);

    ASSERT_EQ(permuteOp->getOpType(), "Permute");
    ASSERT_EQ(permuteOp->attrsCount(), 3);
    ASSERT_EQ(permuteOp->inputSlots(), 1);
    ASSERT_EQ(permuteOp->outputSlots(), 1);
    ASSERT_TRUE(permuteOp->hasTypeTrait("executable"));

}
