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
