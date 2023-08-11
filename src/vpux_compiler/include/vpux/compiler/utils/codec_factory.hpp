//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include <vector>

namespace vpux {

class ICodec {
public:
    enum CompressionAlgorithm {
        HUFFMAN_CODEC,
        BITCOMPACTOR_CODEC,
    };
    virtual std::vector<uint8_t> compress(std::vector<uint8_t>& data) const = 0;
    virtual ~ICodec(){};
};

std::unique_ptr<ICodec> makeCodec(const ICodec::CompressionAlgorithm algo);

}  // namespace vpux
