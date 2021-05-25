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
