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
// Various helper functions to work with C++ enumerations.
//

#pragma once

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace vpux {

//
// EnumSet/EnumMap
//

namespace details {

struct EnumHash final {
    template <typename Enum>
    size_t operator()(Enum val) const {
        return static_cast<size_t>(val);
    }
};

}  // namespace details

template <typename Enum>
using EnumSet = std::unordered_set<Enum, details::EnumHash>;

template <typename Enum, typename Val>
using EnumMap = std::unordered_map<Enum, Val, details::EnumHash>;

}  // namespace vpux
