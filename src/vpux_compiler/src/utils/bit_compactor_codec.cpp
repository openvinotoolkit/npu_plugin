//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef ENABLE_BITCOMPACTOR
#include "bitcompactor/include/bitCompactor.h"
#endif

#include "vpux/compiler/utils/bit_compactor_codec.hpp"

using namespace vpux;

#ifdef ENABLE_BITCOMPACTOR
std::vector<uint8_t> vpux::BitCompactorCodec::compress(std::vector<uint8_t>& data) const {
    VPUX_THROW_WHEN(data.empty(), "BitCompactorCodec::compress: Empty input data vector");

    btc27::BitCompactor bitCompactor;
    btc27::BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    const auto uncompressedDataSize = static_cast<unsigned>(data.size());
    const auto compressedBufferSizeBound = bitCompactor.GetCompressedSizeBound(uncompressedDataSize);

    std::vector<uint8_t> compressedDataBuffer(compressedBufferSizeBound, 0);
    const auto compressedSize = bitCompactor.CompressArray(
            data.data(), uncompressedDataSize, compressedDataBuffer.data(), compressedBufferSizeBound, btcArgs);

    VPUX_THROW_WHEN(compressedSize > compressedBufferSizeBound, "Compression would lead to buffer overflow");

    // Trim trailing bytes.
    compressedDataBuffer.resize(compressedSize);

    // sometimes even if the tensor is > 4KB it might not be compressible
    if (uncompressedDataSize <= compressedSize) {
        return {};
    }

    return compressedDataBuffer;
}
#endif
