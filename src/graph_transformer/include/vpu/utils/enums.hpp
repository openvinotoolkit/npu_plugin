//
// Copyright (C) 2018-2019 Intel Corporation.
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

#pragma once

#include <cstdint>

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <sstream>

#include <vpu/utils/checked_cast.hpp>

namespace vpu {

std::ostream& printValue(std::ostream& os, const std::string& strMap, int32_t val);

#define VPU_DECLARE_ENUM(EnumName, ...)                                         \
    enum class EnumName : int32_t {                                             \
        __VA_ARGS__                                                             \
    };                                                                          \
    inline std::ostream& operator<<(std::ostream& os, EnumName val) {           \
        return vpu::printValue(os, #__VA_ARGS__, static_cast<int32_t>(val));\
    }                                                                           \
    template <typename I>                                                       \
    inline I checked_cast(EnumName val) {                                       \
        return vpu::checked_cast<I>(static_cast<int32_t>(val));                 \
    }

struct EnumClassHash final {
    template <typename E>
    size_t operator()(E t) const {
        return std::hash<int32_t>()(static_cast<int32_t>(t));
    }
};

template <typename E>
using EnumSet = std::unordered_set<E, EnumClassHash>;

template <typename E, typename V>
using EnumMap = std::unordered_map<E, V, EnumClassHash>;

}  // namespace vpu
