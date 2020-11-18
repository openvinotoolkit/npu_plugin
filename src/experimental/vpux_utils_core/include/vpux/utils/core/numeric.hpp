//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
// FP16 type
//

// FP16 storage type for unsupported CPUs.
using fp16_t = short;

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
