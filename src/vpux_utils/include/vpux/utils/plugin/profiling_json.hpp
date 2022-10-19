//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>

namespace vpux {
namespace profiling {

inline int getPrecisionIndex() {
    static int i = std::ios_base::xalloc();
    return i;
}

enum TimeUnitFormat { MS = 1, NS = 0 };  // Microseconds/Nanoseconds

// Stream manipulator to handle different time units. Can be used as
// stream << TimeUnitFormat::MS << event;
// Nanoseconds by default
static inline std::ostream& operator<<(std::ostream& os, const TimeUnitFormat& unit) {
    os.iword(getPrecisionIndex()) = unit;
    return os;
}

// This structure describes a single entry in Tracing Event format.
struct TracingEventDesc {
    std::string name;
    std::string category;
    int pid;
    int tid;
    uint64_t timestamp;
    uint64_t duration;
};

// This stream operator prints TracingEventDesc in JSON format.
// Support "X" event type only - so called "Complete event".
static inline std::ostream& operator<<(std::ostream& os, const struct TracingEventDesc& event) {
    const auto format = static_cast<TimeUnitFormat>(os.iword(getPrecisionIndex()));
    const auto castTime = [&](uint64_t time) {
        if (format == NS) {
            return std::to_string(time);
        }
        return std::to_string(time / 1000.);
    };

    os << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << castTime(event.timestamp) << ", \"dur\":" << castTime(event.duration)
       << ", \"pid\":" << event.pid << ", \"tid\":" << event.tid << ", \"tts\":" << 3 << ", \"args\":{}}," << std::endl;
    return os;
}

}  // namespace profiling
}  // namespace vpux
