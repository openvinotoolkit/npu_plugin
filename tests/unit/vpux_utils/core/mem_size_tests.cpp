//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/mem_size.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_MemSize, ValidCases) {
    const auto bits8 = 8_Bit;
    const auto bytes16 = 16_Byte;
    const auto kilobyte10 = 10_KB;
    const auto megabyte2 = 2_MB;
    const auto gigabyte1 = 1_GB;

    EXPECT_EQ(bits8.count(), 8);
    EXPECT_EQ(bytes16.count(), 16);
    EXPECT_EQ(kilobyte10.count(), 10);
    EXPECT_EQ(megabyte2.count(), 2);
    EXPECT_EQ(gigabyte1.count(), 1);

    EXPECT_EQ(bits8.to<Byte>().count(), 1);
    EXPECT_EQ(bytes16.to<Bit>().count(), 16 * 8);
    EXPECT_EQ(kilobyte10.to<Byte>().count(), 10 * 1024);
    EXPECT_EQ(megabyte2.to<KB>().count(), 2 * 1024);
    EXPECT_EQ(megabyte2.to<Byte>().count(), 2 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.to<MB>().count(), 1 * 1024);
    EXPECT_EQ(gigabyte1.to<KB>().count(), 1 * 1024 * 1024);
    EXPECT_EQ(gigabyte1.to<Byte>().count(), 1 * 1024 * 1024 * 1024);
}
