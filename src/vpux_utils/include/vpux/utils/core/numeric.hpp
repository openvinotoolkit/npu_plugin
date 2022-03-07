//
// Copyright 2020 Intel Corporation.
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

//
// Various numeric helper functions.
//

#pragma once

#include "vpux/utils/core/type_traits.hpp"

#include <limits>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace vpux {

//
// Integer arithmetic
//

template <typename T, typename = require_t<std::is_integral<T>>>
T isPowerOfTwo(T val) {
    return (val > 0) && ((val & (val - 1)) == 0);
}

template <typename T, typename = require_t<std::is_integral<T>>>
T alignVal(T val, T align) {
    return (val + (align - 1)) & ~(align - 1);
}

template <typename T, typename = require_t<std::is_integral<T>>>
T divUp(T a, T b) {
    return (a + b - 1) / b;
}

//
// Float arithmetic
//

inline bool isFloatEqual(float a, float b) {
    return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}
inline bool isDoubleEqual(double a, double b) {
    return std::fabs(a - b) <= std::numeric_limits<double>::epsilon();
}

}  // namespace vpux
