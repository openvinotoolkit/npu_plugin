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
#include <llvm/Support/WithColor.h>

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

Logger vpux::Logger::nest() const {
    return nest(name());
}

Logger vpux::Logger::nest(StringLiteral name) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + 1;
    return nested;
}

namespace {

llvm::raw_ostream& getStream(LogLevel msgLevel) {
    switch (msgLevel) {
    case LogLevel::Fatal:
    case LogLevel::Error:
        return llvm::WithColor(llvm::outs(), llvm::HighlightColor::Error).get();
    case LogLevel::Warning:
        return llvm::WithColor(llvm::outs(), llvm::HighlightColor::Warning).get();
    case LogLevel::Info:
        return llvm::WithColor(llvm::outs(), llvm::HighlightColor::Note).get();
    case LogLevel::Debug:
    case LogLevel::Trace:
        return llvm::WithColor(llvm::outs(), llvm::HighlightColor::Remark).get();
    default:
        return llvm::outs();
    }
}

}  // namespace

void vpux::Logger::addEntryPacked(LogLevel msgLevel, const llvm::formatv_object_base& msg) const {
#ifdef NDEBUG
    if (!isActive(msgLevel)) {
        return;
    }
#else
    if (!isActive(msgLevel) && !(llvm::DebugFlag && ::llvm::isCurrentDebugType(name().data()))) {
        return;
    }
#endif

    auto& stream = getStream(msgLevel);

    printTo(stream, "[{0}] ", msgLevel);

    for (size_t i = 0; i < _indentLevel; ++i)
        stream << "  ";

    msg.format(stream);
    stream << "\n";

    stream.flush();
}
