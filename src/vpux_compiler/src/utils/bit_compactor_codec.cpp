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

#ifdef ENABLE_BITCOMPACTOR

#include "vpux/compiler/utils/bit_compactor_codec.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

vpux::BitCompactorCodec::BitCompactorCodec(): _bitCompactor() {
    _bitCompactor.mBitCompactorConfig->blockSize = 64;
    _bitCompactor.mBitCompactorConfig->superBlockSize = 4096;
    _bitCompactor.mBitCompactorConfig->minFixedBitLn = 3;
    _bitCompactor.mBitCompactorConfig->cmprs = 1;
    _bitCompactor.mBitCompactorConfig->bypass_en = false;
    _bitCompactor.mBitCompactorConfig->dual_encode_en = true;
    _bitCompactor.mBitCompactorConfig->proc_bin_en = false;
    _bitCompactor.mBitCompactorConfig->proc_btmap_en = false;
    _bitCompactor.mBitCompactorConfig->mixedBlkSize = false;
    _bitCompactor.mBitCompactorConfig->align = 1;
    _bitCompactor.mBitCompactorConfig->ratio = false;
    _bitCompactor.mBitCompactorConfig->verbosity = 0;  // set between 0-5,
                                                       // 0 shows basic info,
                                                       // 3 shows Metadata and some other useful stuff,
                                                       // 5 shows all available info
}

std::vector<uint8_t> vpux::BitCompactorCodec::compress(std::vector<uint8_t>& data) const {
    VPUX_THROW_WHEN(data.empty(), "BitCompactorCodec::compress: Empty input data vector");

    BitCompactor::btcmpctr_compress_wrap_args_t btcArgs;

    btcArgs.bypass_en = _bitCompactor.mBitCompactorConfig->bypass_en;
    btcArgs.dual_encode_en = _bitCompactor.mBitCompactorConfig->dual_encode_en;
    btcArgs.proc_bin_en = _bitCompactor.mBitCompactorConfig->proc_bin_en;
    btcArgs.proc_btmap_en = _bitCompactor.mBitCompactorConfig->proc_btmap_en;
    btcArgs.align = _bitCompactor.mBitCompactorConfig->align;
    btcArgs.verbosity = _bitCompactor.mBitCompactorConfig->verbosity;
    btcArgs.SblkSize = _bitCompactor.mBitCompactorConfig->blockSize;
    btcArgs.LblkSize = _bitCompactor.mBitCompactorConfig->superBlockSize;
    btcArgs.mixedBlkSize = _bitCompactor.mBitCompactorConfig->mixedBlkSize;
    btcArgs.minFixedBitLn = _bitCompactor.mBitCompactorConfig->minFixedBitLn;

    const auto uncompressedDataSize = static_cast<int32_t>(data.size());
    const auto compressedBufferSizeBound = _bitCompactor.btcmpctr_cmprs_bound(uncompressedDataSize);

    std::vector<uint8_t> compressedDataBuffer(compressedBufferSizeBound, 0);
    const auto compressedSize = _bitCompactor.CompressArray(
            data.data(), uncompressedDataSize, compressedDataBuffer.data(), compressedBufferSizeBound, &btcArgs);
    // Trim trailing bytes.
    compressedDataBuffer.resize(compressedSize);

    // sometimes even if the tensor is > 4KB it might not be compressable
    if (uncompressedDataSize <= compressedSize) {
        return {};
    }

    return compressedDataBuffer;
}

#endif
