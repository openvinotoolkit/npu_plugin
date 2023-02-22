//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>
#include <climits>
#include "vpux/compiler/utils/codec_factory.hpp"

using namespace vpux;

using MLIR_CompressionTest = testing::Test;

namespace {
uint16_t crc16(const std::vector<uint8_t>& inputBytes) {
    // g(x) = x ^ 16 + x ^ 13 + x ^ 12 + x ^ 11 + x ^ 10 + x ^ 8 + x ^ 6 + x ^ 2 + x + 1
    const uint16_t polynomial = 0x3d65;
    // For int16 0xff00 are bytes of interest, hence the shift is 2 bytes * 8 bits - 8 bits
    const uint16_t highByteShift = (sizeof(uint16_t) * CHAR_BIT) - CHAR_BIT;
    // 0x8000 can be calculated as 1 << 15, i.e. 1 << (2 bytes * 8 bits - 1)
    const uint16_t msbShift = (sizeof(uint16_t) * CHAR_BIT) - 1;
    // MSB stands for 'most significant bit'
    const uint16_t msbMask = 1 << msbShift;
    uint16_t remainder = 0xffff;
    for (const uint8_t& byteValue : inputBytes) {
        remainder = remainder ^ (byteValue << highByteShift);
        for (size_t bitPos = 0; bitPos < CHAR_BIT; bitPos++) {
            if (remainder & msbMask) {
                remainder = (remainder << 1) ^ polynomial;
            } else {
                remainder = remainder << 1;
            }
        }
    }

    return remainder;
}
};  // namespace

TEST_F(MLIR_CompressionTest, compressSingleArrayMultipleTimes) {
    const auto codec = vpux::makeCodec(ICodec::CompressionAlgorithm::HUFFMAN_CODEC);
    std::vector<uint16_t> crcList;
    // 65536 bytes because initial reproducer had that much data.
    const size_t dataSize = 65536;
    // 71 is the largest supersingular prime number.
    std::vector<uint8_t> origData(dataSize, 71);
    const size_t repeats = 4;
    for (size_t iter = 0; iter < repeats; iter++) {
        const auto compressedData = codec->compress(origData);
        ASSERT_FALSE(compressedData.empty());
        crcList.push_back(crc16(compressedData));
    }
    const auto comparePredicate = [crcList](const uint16_t& val) -> bool {
        return val == 0x54e3;
    };
    ASSERT_TRUE(std::all_of(crcList.cbegin(), crcList.cend(), comparePredicate));
}

TEST_F(MLIR_CompressionTest, compressDifferentArrays) {
    const auto codec = vpux::makeCodec(ICodec::CompressionAlgorithm::HUFFMAN_CODEC);
    // 65536 twice just for the sake of variety.
    const size_t dataSize = 65536 * 2;
    std::map<uint8_t, uint16_t> valuesAndSums = {
            {2, 0xc2e5},  {3, 0x97cd},  {5, 0xb934},  {7, 0x1364},  {11, 0xa3fa},
            {13, 0x8d03}, {17, 0x55f3}, {19, 0xffa3}, {23, 0x7b0a}, {29, 0xe56d},
            {31, 0x4f3d}, {41, 0xd976}, {47, 0xf78f}, {59, 0x1b48}, {71, 0x8fb9},
    };

    for (const auto& value : valuesAndSums) {
        std::vector<uint8_t> origData(dataSize, value.first);
        const auto compressedData = codec->compress(origData);
        ASSERT_FALSE(compressedData.empty());
        ASSERT_EQ(crc16(compressedData), value.second);
    }
}

TEST_F(MLIR_CompressionTest, compressSparseMatrix) {
    const auto codec = vpux::makeCodec(ICodec::CompressionAlgorithm::HUFFMAN_CODEC);
    const size_t rowCount = 256;
    const size_t colCount = 256;
    const size_t dataSize = rowCount * colCount;
    std::vector<uint8_t> origData(dataSize, 0);
    for (size_t diagPos = 0; diagPos < rowCount && diagPos < colCount; diagPos++) {
        size_t matrixOffset = diagPos * colCount + diagPos;
        origData.at(matrixOffset) = matrixOffset % 256;
    }

    std::vector<uint16_t> crcList;
    const size_t repeats = 4;
    for (size_t iter = 0; iter < repeats; iter++) {
        const auto compressedData = codec->compress(origData);
        ASSERT_FALSE(compressedData.empty());
        crcList.push_back(crc16(compressedData));
    }
    const auto comparePredicate = [crcList](const uint16_t& val) -> bool {
        return val == 0xfc7d;
    };
    ASSERT_TRUE(std::all_of(crcList.cbegin(), crcList.cend(), comparePredicate));
}
