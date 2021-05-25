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

#include "vpux/utils/core/checked_cast.hpp"

#include <ie_common.h>
#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_CheckedCast, ValidCases) {
    const int8_t valI8 = 10;
    const int16_t valI16 = 10;
    const int32_t valI32 = 10;
    const int64_t valI64 = 10;

    const uint8_t valU8 = 10;
    const uint16_t valU16 = 10;
    const uint32_t valU32 = 10;
    const uint64_t valU64 = 10;

    const float valF32 = 10.0f;
    const double valF64 = 10.0;

    const auto test = [](auto val) -> void {
        EXPECT_EQ(checked_cast<int8_t>(val), static_cast<int8_t>(val));
        EXPECT_EQ(checked_cast<int16_t>(val), static_cast<int16_t>(val));
        EXPECT_EQ(checked_cast<int32_t>(val), static_cast<int32_t>(val));
        EXPECT_EQ(checked_cast<int64_t>(val), static_cast<int64_t>(val));

        EXPECT_EQ(checked_cast<uint8_t>(val), static_cast<uint8_t>(val));
        EXPECT_EQ(checked_cast<uint16_t>(val), static_cast<uint16_t>(val));
        EXPECT_EQ(checked_cast<uint32_t>(val), static_cast<uint32_t>(val));
        EXPECT_EQ(checked_cast<uint64_t>(val), static_cast<uint64_t>(val));
    };

    test(valI8);
    test(valI16);
    test(valI32);
    test(valI64);

    test(valU8);
    test(valU16);
    test(valU32);
    test(valU64);

    test(valF32);
    test(valF64);
}

TEST(MLIR_CheckedCast, InvalidThrowsException) {
    using Exception = InferenceEngine::Exception;

    const int8_t valI8 = -10;
    const int16_t valI16 = -10;
    const int32_t valI32 = -10;
    const int64_t valI64 = -10;

    const float valF32 = -10.0f;
    const double valF64 = -10.0;

    const auto test = [](auto val) -> void {
        EXPECT_THROW(checked_cast<uint8_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint16_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint32_t>(val), Exception);
        EXPECT_THROW(checked_cast<uint64_t>(val), Exception);
    };

    test(valI8);
    test(valI16);
    test(valI32);
    test(valI64);

    test(valF32);
    test(valF64);
}
