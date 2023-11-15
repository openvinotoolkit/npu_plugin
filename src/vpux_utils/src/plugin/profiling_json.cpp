//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/plugin/profiling_json.hpp"

using namespace vpux::profiling;

namespace vpux {
namespace profiling {

std::ostream& operator<<(std::ostream& os, const TraceEventDesc& event) {
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

    os << "}";
    os.flags(origFlags);
    return os;
}

TraceEventTimeOrderedDistribution::TraceEventTimeOrderedDistribution(unsigned startThreadId)
        : _startThreadId(startThreadId) {
}

unsigned TraceEventTimeOrderedDistribution::getThreadId(double taskStartTime, double taskEndTime) {
    if (size() == 0) {
        push_back(taskEndTime);
        _threadId.push_back(_startThreadId);
        return _startThreadId;
    }

    unsigned thId = _startThreadId;
    for (auto& prevEndTime : *this) {
        if (prevEndTime <= taskStartTime) {
            prevEndTime = taskEndTime;  // update occupancy in this thread with the end time of the current task
            return thId;
        }
        thId++;
    }
    push_back(taskEndTime);
    _threadId.push_back(thId);
    return thId;
}

void TraceEventTimeOrderedDistribution::setThreadIdOffset(unsigned offset) {
    _startThreadId = offset;
}

std::vector<unsigned> TraceEventTimeOrderedDistribution::getThreadIds() const {
    return _threadId;
}

}  // namespace profiling
}  // namespace vpux
