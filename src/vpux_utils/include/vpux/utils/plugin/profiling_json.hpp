//
// Copyright Intel Corporation.
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
