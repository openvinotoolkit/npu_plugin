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
