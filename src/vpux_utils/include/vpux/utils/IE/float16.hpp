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
