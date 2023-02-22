//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <iomanip>
#include <ostream>
#include <string>

namespace vpux {
namespace profiling {

// This structure describes a single entry in Tracing Event format.
struct TracingEventDesc {
    std::string name;
    std::string category;
    int pid;
    int tid;
    // Trace Event Format expects timestamp and duration in microseconds but it may be fractional
    // JSON numbers are doubles anyways so use double for more flexibility
    double timestamp;
    double duration;
};

// This stream operator prints TracingEventDesc in JSON format.
// Support "X" event type only - so called "Complete event".
static inline std::ostream& operator<<(std::ostream& os, const struct TracingEventDesc& event) {
    os << std::fixed << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << event.timestamp << ", \"dur\":" << event.duration << ", \"pid\":" << event.pid
       << ", \"tid\":" << event.tid << ", \"args\":{}}," << std::endl;
    return os;
}

}  // namespace profiling
}  // namespace vpux
