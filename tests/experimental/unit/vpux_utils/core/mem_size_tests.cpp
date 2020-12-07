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

#include "vpux/utils/core/mem_size.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(MemSize, ValidCases) {
    const auto bytes16 = 16_Byte;
    const auto kilobyte10 = 10_KB;
    const auto megabyte2 = 2_MB;
    const auto gigabyte1 = 1_GB;

    EXPECT_EQ(bytes16.count(), 16);
    EXPECT_EQ(kilobyte10.count(), 10);
    EXPECT_EQ(kilobyte10.to<MemType::Byte>().count(), 10 * 1024);
    EXPECT_EQ(megabyte2.count(), 2);
    EXPECT_EQ(megabyte2.to<MemType::KB>().count(), 2 * 1024);
    EXPECT_EQ(megabyte2.to<MemType::Byte>().count(), 2 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.count(), 1);
    EXPECT_EQ(gigabyte1.to<MemType::MB>().count(), 1 * 1024);
    EXPECT_EQ(gigabyte1.to<MemType::KB>().count(), 1 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.to<MemType::Byte>().count(), 1 * 1024 * 1024 * 1024);
}
