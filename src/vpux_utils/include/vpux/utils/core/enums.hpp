//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
