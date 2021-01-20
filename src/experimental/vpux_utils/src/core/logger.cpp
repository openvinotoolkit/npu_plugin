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

#include "vpux/utils/core/logger.hpp"

#include <llvm/Support/Debug.h>

#include <cstdio>

using namespace vpux;

//
// LogLevel
//

StringLiteral vpux::stringifyEnum(LogLevel val) {
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

//
// Logger
//

Logger& vpux::Logger::global() {
#ifdef NDEBUG
    static Logger log("global", LogLevel::None);
#else
    static Logger log("global", LogLevel::Warning);
#endif

    return log;
}

vpux::Logger::Logger(StringLiteral name, LogLevel lvl): _name(name), _logLevel(lvl) {
}

Logger vpux::Logger::nest(size_t inc) const {
    return nest(name(), inc);
}

Logger vpux::Logger::nest(StringLiteral name, size_t inc) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + inc;
    return nested;
}

bool vpux::Logger::isActive(LogLevel msgLevel) const {
#ifdef NDEBUG
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel);
#else
    return (static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel)) ||
           (llvm::DebugFlag && llvm::isCurrentDebugType(name().data()));
#endif
}

llvm::raw_ostream& vpux::Logger::getBaseStream() {
#ifdef NDEBUG
    return llvm::outs();
#else
    return llvm::DebugFlag ? llvm::dbgs() : llvm::outs();
#endif
}

namespace {

llvm::raw_ostream::Colors getColor(LogLevel msgLevel) {
    switch (msgLevel) {
    case LogLevel::Fatal:
    case LogLevel::Error:
        return llvm::raw_ostream::RED;
    case LogLevel::Warning:
        return llvm::raw_ostream::YELLOW;
    case LogLevel::Info:
        return llvm::raw_ostream::CYAN;
    case LogLevel::Debug:
    case LogLevel::Trace:
        return llvm::raw_ostream::GREEN;
    default:
        return llvm::raw_ostream::SAVEDCOLOR;
    }
}

}  // namespace

llvm::WithColor vpux::Logger::getLevelStream(LogLevel msgLevel) {
    const auto color = getColor(msgLevel);
    return llvm::WithColor(getBaseStream(), color, true, false, llvm::ColorMode::Auto);
}

void vpux::Logger::addEntryPacked(LogLevel msgLevel, const llvm::formatv_object_base& msg) const {
    if (!isActive(msgLevel)) {
        return;
    }

    auto colorStream = getLevelStream(msgLevel);
    auto& stream = colorStream.get();

    printTo(stream, "[{0}] ", _name);

    for (size_t i = 0; i < _indentLevel; ++i)
        stream << "  ";

    msg.format(stream);
    stream << "\n";

    stream.flush();
}
