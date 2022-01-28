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

#include "vpux/utils/core/logger.hpp"

#include "vpux/utils/core/optional.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cstdio>
#include <mutex>

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
#ifdef VPUX_DEVELOPER_BUILD
    static Logger log("global", LogLevel::Warning);
#else
    static Logger log("global", LogLevel::None);
#endif

    return log;
}

vpux::Logger::Logger(StringLiteral name, LogLevel ): _name(name), _logLevel(LogLevel::Trace) {
}

Logger vpux::Logger::nest(size_t inc) const {
    return nest(name(), inc);
}

Logger vpux::Logger::nest(StringLiteral name, size_t inc) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + inc;
    return nested;
}

Logger vpux::Logger::unnest(size_t inc) const {
    assert(_indentLevel >= inc);
    Logger unnested(name(), level());
    unnested._indentLevel = _indentLevel - inc;
    return unnested;
}

bool vpux::Logger::isActive(LogLevel ) const {
    return true;
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

    llvm::SmallString<512> tempBuf;
    llvm::raw_svector_ostream tempStream(tempBuf);

    printTo(tempStream, "[{0}] ", _name);

    for (size_t i = 0; i < _indentLevel; ++i)
        tempStream << "  ";

    msg.format(tempStream);
    tempStream << "\n";

    static std::mutex logMtx;
    std::lock_guard<std::mutex> logMtxLock(logMtx);

    auto colorStream = getLevelStream(msgLevel);
    auto& stream = colorStream.get();
    stream << tempStream.str();
    stream.flush();
}
