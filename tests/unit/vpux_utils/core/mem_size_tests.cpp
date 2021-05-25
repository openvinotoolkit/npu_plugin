//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
