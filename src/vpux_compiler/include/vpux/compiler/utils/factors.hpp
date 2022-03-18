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

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Block.h>

namespace vpux {

struct Factors final {
    int64_t larger = 0;
    int64_t smaller = 0;

    Factors() {
    }
    Factors(int64_t larger, int64_t smaller): larger(larger), smaller(smaller) {
    }
};

SmallVector<Factors> getFactorsList(int64_t n);
}  // namespace vpux
