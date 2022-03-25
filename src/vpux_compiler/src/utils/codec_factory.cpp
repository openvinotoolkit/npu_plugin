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
