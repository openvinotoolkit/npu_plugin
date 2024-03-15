//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// FP16 and BF16 implementation
//

#pragma once

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

namespace vpux {

using ov::bfloat16;
using ov::float16;

template <typename OutT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(ov::bfloat16 val) {
    return float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(ov::float16 val) {
    return bfloat16(static_cast<float>(val));
}

template <typename OutT>
enable_t<OutT, not_<std::is_same<ov::float16, OutT>>> checked_cast(ov::bfloat16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<std::is_same<ov::bfloat16, OutT>>> checked_cast(ov::float16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(InT val) {
    return ov::bfloat16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(InT val) {
    return ov::float16(checked_cast<float>(val));
}

}  // namespace vpux
