//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/common_logger.hpp"

namespace vpux {
//
// LogLevel
//

std::string_view stringifyEnum(LogLevel val) {
    switch (val) {
    case LogLevel::None:
        return "None";
    case LogLevel::Fatal:
        return "Fatal";
    case LogLevel::Error:
        return "Error";
    case LogLevel::Warning:
        return "Warning";
    case LogLevel::Info:
        return "Info";
    case LogLevel::Debug:
        return "Debug";
    case LogLevel::Trace:
        return "Trace";
    default:
        return "<UNKNOWN>";
    }
}
}  // namespace vpux
