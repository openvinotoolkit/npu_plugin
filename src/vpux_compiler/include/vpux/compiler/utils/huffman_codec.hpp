//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <vector>
#include "huffmanCodec.hpp"
#include "vpux/compiler/utils/codec_factory.hpp"

namespace vpux {

class HuffmanCodec final : public ICodec {
public:
    HuffmanCodec();
    ~HuffmanCodec() = default;
    HuffmanCodec(const HuffmanCodec&) = delete;
    HuffmanCodec(const HuffmanCodec&&) = delete;
    HuffmanCodec& operator=(const HuffmanCodec&) = delete;
    HuffmanCodec& operator=(const HuffmanCodec&&) = delete;
    std::vector<uint8_t> compress(std::vector<uint8_t>& data) const;

private:
    mutable huffmanCodec _huffmanCodec;
};

}  // namespace vpux
