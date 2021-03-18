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
