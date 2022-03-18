//
// Copyright Intel Corporation.
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

#include "vpux/compiler/utils/huffman_codec.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

constexpr uint32_t bitPerSymbol = 8;
constexpr uint32_t maxNumberEncodedSymbols = 16;
constexpr uint32_t verbosity = 0;
constexpr uint32_t blockSize = 4096;
constexpr bool pStatsOnly = false;
constexpr uint32_t bypassMode = 0;

vpux::HuffmanCodec::HuffmanCodec()
        : _huffmanCodec(bitPerSymbol, maxNumberEncodedSymbols, verbosity, blockSize, pStatsOnly, bypassMode) {
}

std::vector<uint8_t> vpux::HuffmanCodec::compress(std::vector<uint8_t>& data) const {
    VPUX_THROW_WHEN(data.empty(), "HuffmanCodec::compress: Empty input data vector");

    uint32_t uncompressedDataSize = static_cast<int32_t>(data.size());
    const auto compressedBufferSizeBound =
            static_cast<int32_t>(uncompressedDataSize + 2 * (std::ceil(uncompressedDataSize / 4096.0) + 1));

    std::vector<uint8_t> compressedDataBuffer(compressedBufferSizeBound, 0);
    const auto compressedSize =
            _huffmanCodec.huffmanCodecCompressArray(uncompressedDataSize, data.data(), compressedDataBuffer.data());

    // Trim trailing bytes.
    compressedDataBuffer.resize(compressedSize);

    // sometimes even if the tensor is > 4KB it might not be compressible
    if (uncompressedDataSize <= compressedSize) {
        return {};
    }

    return compressedDataBuffer;
}
