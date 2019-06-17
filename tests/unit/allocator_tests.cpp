//
// Copyright 2019 Intel Corporation.
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

#include <gtest/gtest.h>
#include "kmb_allocator.h"

using namespace testing;
using namespace vpu::KmbPlugin;

class kmbAllocatorUnitTests : public ::testing::Test {

};

TEST(kmbAllocatorUnitTests, canAllocateMemory) {
    KmbAllocator allocator;

    size_t size = 10;
    ASSERT_NE(allocator.alloc(size), nullptr);
}