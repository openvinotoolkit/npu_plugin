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
