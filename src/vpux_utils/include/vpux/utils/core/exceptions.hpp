//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <stdexcept>
#include "vpux/utils/core/common_string_utils.hpp"

namespace vpux {
class CoreException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace details {

[[noreturn]] void throwNonFormated(const char* file, int line, const std::string& str);

}
}  // namespace vpux

#define CORE_VPUX_THROW(...) vpux::details::throwNonFormated(__FILE__, __LINE__, vpux::printFormattedCStr(__VA_ARGS__))

#define CORE_VPUX_THROW_UNLESS(_condition_, ...) \
    if (!(_condition_))                          \
    CORE_VPUX_THROW(__VA_ARGS__)

#define CORE_VPUX_THROW_WHEN(_condition_, ...) \
    if ((_condition_))                         \
    CORE_VPUX_THROW(__VA_ARGS__)
