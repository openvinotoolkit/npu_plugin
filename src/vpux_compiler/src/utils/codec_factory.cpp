//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include "vpux/compiler/utils/bit_compactor_codec.hpp"
#include "vpux/compiler/utils/huffman_codec.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

std::unique_ptr<ICodec> makeCodec(const ICodec::CompressionAlgorithm algo) {
    switch (algo) {
    case ICodec::CompressionAlgorithm::HUFFMAN_CODEC:
        return std::make_unique<vpux::HuffmanCodec>();
    case ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC:
#ifdef ENABLE_BITCOMPACTOR
        return std::make_unique<vpux::BitCompactorCodec>();
#else
        VPUX_THROW("vpux::makeCodec: bitcompactor is disabled");
#endif
    }
    VPUX_THROW("vpux::makeCodec: unsupported compression algorithm");
}

}  // namespace vpux
