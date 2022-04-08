//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

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
    uint64_t timestamp;
    uint64_t duration;
};

// This stream operator prints TracingEventDesc in JSON format.
// Support "X" event type only - so called "Complete event".
static inline std::ostream& operator<<(std::ostream& os, const struct TracingEventDesc& event) {
    os << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << event.timestamp << ", \"dur\":" << event.duration << ", \"pid\":" << event.pid
       << ", \"tid\":" << event.tid << ", \"tts\":" << 3 << ", \"args\":{}}," << std::endl;
    return os;
}

}  // namespace profiling
}  // namespace vpux
