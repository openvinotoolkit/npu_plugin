//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

//

#include "vpux/utils/plugin/profiling_json.hpp"

using namespace vpux::profiling;

namespace vpux {
namespace profiling {

std::ostream& operator<<(std::ostream& os, const TracingEventDesc& event) {
    std::ios::fmtflags origFlags(os.flags());

    os << std::fixed << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << event.timestamp << ", \"dur\":" << event.duration << ", \"pid\":" << event.pid
       << ", \"tid\":" << event.tid;

    if (!event.customArgs.empty()) {
        os << ", \"args\":{";
        bool isFirst = true;
        for (auto const& arg : event.customArgs) {
            os << (isFirst ? "" : ", ") << "\"" << arg.first << "\": \"" << arg.second << "\"";
            isFirst = false;
        }
        os << "}";
    }

    os << "}," << std::endl;
    os.flags(origFlags);
    return os;
}

}  // namespace profiling
}  // namespace vpux
