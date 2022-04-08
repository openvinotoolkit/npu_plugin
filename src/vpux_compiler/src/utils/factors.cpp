//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/factors.hpp"

using namespace vpux;

SmallVector<Factors> vpux::getFactorsList(int64_t n) {
    SmallVector<Factors> factors;
    for (int64_t i = 1; i <= sqrt(n); i++) {
        if (n % i == 0) {
            factors.emplace_back(n / i, i);  // larger, smaller
        }
    }
    return factors;
}

SmallVector<Factors> vpux::getFactorsListWithLimitation(int64_t n, int64_t limit) {
    SmallVector<Factors> factors;
    for (int64_t i = n; i >= 1; i--) {
        if ((n % i == 0) && ((n / i) <= limit)) {
            factors.emplace_back(n / i, i);  // larger, smaller
        }
    }
    return factors;
}
