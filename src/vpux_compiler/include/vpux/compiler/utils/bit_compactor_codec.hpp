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

#ifdef ENABLE_BITCOMPACTOR

#include <vector>
#include "bitCompactor.h"
#include "vpux/compiler/utils/codec_factory.hpp"

namespace vpux {

class BitCompactorCodec final : public ICodec {
public:
    BitCompactorCodec();
    ~BitCompactorCodec() = default;
    BitCompactorCodec(const BitCompactorCodec&) = delete;
    BitCompactorCodec(const BitCompactorCodec&&) = delete;
    BitCompactorCodec& operator=(const BitCompactorCodec&) = delete;
    BitCompactorCodec& operator=(const BitCompactorCodec&&) = delete;
    std::vector<uint8_t> compress(std::vector<uint8_t>& data) const;

private:
    // Classified mutable because BitCompactor::btcmpctr_cmprs_bound is non-constant.
    // FIXME Make the method constant in bitcompactor repository.
    mutable BitCompactor _bitCompactor;
};

}  // namespace vpux

#endif
