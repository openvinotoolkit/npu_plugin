//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <memory>
#include <vector>

namespace vpux {

class ICodec {
public:
    enum CompressionAlgorithm { HUFFMAN_CODEC, BITCOMPACTOR_CODEC };
    enum class CompressionMode { UINT8, FP16 };
    virtual bool supportsFP16compression() const;
    virtual mlir::FailureOr<std::vector<uint8_t>> compress(std::vector<uint8_t>& data,
                                                           CompressionMode mode = CompressionMode::UINT8,
                                                           const Logger& _log = vpux::Logger::global()) const = 0;
    virtual ~ICodec(){};

    static std::string compressionModeToStr(ICodec::CompressionMode mode);
};

std::unique_ptr<ICodec> makeCodec(const ICodec::CompressionAlgorithm algo, VPU::ArchKind arch = VPU::ArchKind::UNKNOWN);
}  // namespace vpux
