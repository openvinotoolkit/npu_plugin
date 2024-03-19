//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include "vpux/utils/core/common_logger.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

namespace vpux {
class Logger;
struct LoggerAdapter {
    LoggerAdapter(std::string_view category);
    LoggerAdapter();
    ~LoggerAdapter();

    template <class... Args>
    void trace(const char* fmt, Args&&... args) {
        printImpl(LogLevel::Trace, printFormattedCStr(fmt, std::forward<Args>(args)...));
    }

    template <class... Args>
    void info(const char* fmt, Args&&... args) {
        printImpl(LogLevel::Info, printFormattedCStr(fmt, std::forward<Args>(args)...));
    }

    template <class... Args>
    void warning(const char* fmt, Args&&... args) {
        printImpl(LogLevel::Warning, printFormattedCStr(fmt, std::forward<Args>(args)...));
    }

    static void setGlobalLevel(LogLevel lvl);

private:
    void printImpl(LogLevel level, const std::string& str);
    std::unique_ptr<Logger> impl;
};
}  // namespace vpux
