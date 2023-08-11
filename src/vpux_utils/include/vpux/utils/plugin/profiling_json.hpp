//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

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
    std::vector<std::pair<std::string, std::string>> customArgs;
};

// This stream operator prints TracingEventDesc in JSON format.
// Support Tracing Event Format's "X" event type only - so called "Complete event".
std::ostream& operator<<(std::ostream& os, const TracingEventDesc& event);
}  // namespace profiling
}  // namespace vpux
