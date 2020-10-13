//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/core/dims_order.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(DimsOrderTest, Named) {
    EXPECT_EQ(DimsOrder::C.numDims(), 1);

    EXPECT_EQ(DimsOrder::NC.numDims(), 2);

    EXPECT_EQ(DimsOrder::CHW.numDims(), 3);
    EXPECT_EQ(DimsOrder::HWC.numDims(), 3);
    EXPECT_EQ(DimsOrder::HCW.numDims(), 3);

    EXPECT_EQ(DimsOrder::NCHW.numDims(), 4);
    EXPECT_EQ(DimsOrder::NHWC.numDims(), 4);
    EXPECT_EQ(DimsOrder::NHCW.numDims(), 4);

    EXPECT_EQ(DimsOrder::NCDHW.numDims(), 5);
    EXPECT_EQ(DimsOrder::NDHWC.numDims(), 5);
}

TEST(DimsOrderTest, MemDim_4D) {
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(0)), MemDim(3));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(1)), MemDim(2));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(2)), MemDim(1));
    EXPECT_EQ(DimsOrder::NCHW.toMemDim(Dim(3)), MemDim(0));

    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(0)), MemDim(3));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(1)), MemDim(0));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(2)), MemDim(2));
    EXPECT_EQ(DimsOrder::NHWC.toMemDim(Dim(3)), MemDim(1));
}
