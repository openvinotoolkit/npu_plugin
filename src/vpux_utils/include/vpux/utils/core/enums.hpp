//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Various helper functions to work with C++ enumerations.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/type_traits.hpp"

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

//
// `stringifyEnum` function handling.
//

namespace details {

template <class Enum, typename = void>
struct HasStringifyEnum {
    static constexpr bool value = false;
};

template <class Enum>
struct HasStringifyEnum<Enum, require_t<std::is_base_of<StringRef, decltype(stringifyEnum(std::declval<Enum>()))>>> {
    static constexpr bool value = true;
};

}  // namespace details

}  // namespace vpux

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename Enum>
struct format_provider<Enum, vpux::require_t<std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>> final {
    static void format(const Enum& val, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(stringifyEnum(val)).format(stream, style);
    }
};

}  // namespace llvm
