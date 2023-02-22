//
// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
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
SmallVector<Factors> getFactorsListWithLimitation(int64_t n, int64_t limit);
}  // namespace vpux
