//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// FP16 and BF16 implementation
//

#pragma once

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>

namespace vpux {

using ngraph::bfloat16;
using ngraph::float16;

template <typename OutT>
enable_t<OutT, std::is_same<ngraph::float16, OutT>> checked_cast(ngraph::bfloat16 val) {
    return float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ngraph::bfloat16, OutT>> checked_cast(ngraph::float16 val) {
    return bfloat16(static_cast<float>(val));
}

template <typename OutT>
enable_t<OutT, not_<std::is_same<ngraph::float16, OutT>>> checked_cast(ngraph::bfloat16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<std::is_same<ngraph::bfloat16, OutT>>> checked_cast(ngraph::float16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ngraph::bfloat16, OutT>> checked_cast(InT val) {
    return ngraph::bfloat16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ngraph::float16, OutT>> checked_cast(InT val) {
    return ngraph::float16(checked_cast<float>(val));
}

}  // namespace vpux
