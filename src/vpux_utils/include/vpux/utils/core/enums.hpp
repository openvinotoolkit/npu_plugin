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
// Various helper functions to work with C++ enumerations.
//

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <cstddef>

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
