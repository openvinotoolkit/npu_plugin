//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/type_traits.hpp"

#include <string_view>

namespace vpux {
//
// LogLevel
//

// Logger verbosity levels

enum class LogLevel {
    None = 0,     // Logging is disabled
    Fatal = 1,    // Used for very severe error events that will most probably
                  // cause the application to terminate
    Error = 2,    // Reporting events which are not expected during normal
                  // execution, containing probable reason
    Warning = 3,  // Indicating events which are not usual and might lead to
                  // errors later
    Info = 4,     // Short enough messages about ongoing activity in the process
    Debug = 5,    // More fine-grained messages with references to particular data
                  // and explanations
    Trace = 6,    // Involved and detailed information about execution, helps to
                  // trace the execution flow, produces huge output
};

std::string_view stringifyEnum(LogLevel val);
}  // namespace vpux

namespace vpux {
TYPE_PRINTER(vpux::LogLevel)
}  // namespace vpux
