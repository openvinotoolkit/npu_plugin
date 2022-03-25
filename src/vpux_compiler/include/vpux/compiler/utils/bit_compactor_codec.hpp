//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
