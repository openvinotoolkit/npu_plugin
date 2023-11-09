//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace vpux {
namespace profiling {

// This structure describes a single entry in Tracing Event format.
struct TraceEventDesc {
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

// This stream operator prints TraceEventDesc in JSON format.
// Support Tracing Event Format's "X" event type only - so called "Complete event".
std::ostream& operator<<(std::ostream& os, const TraceEventDesc& event);

/**
 * @brief Helper class to calculate placement of profiling tasks in
 * an optimal number of Perfetto UI threads
 *
 * Stores tasks end times for each thread.
 */
class TraceEventTimeOrderedDistribution : private std::vector<double> {
public:
    TraceEventTimeOrderedDistribution(unsigned startThreadId = 0);
    /**
     * @brief Get the event thread Id assuring a non-overlapping placement among other existing tasks on the same
     * thread.
     *
     * Calls to this function assume that tasks were sorted in ascending order by taskStartTime
     * This function updates the state of object with the taskEndTime in the thread the task was assigned to.
     *
     * @return int - calculate thread id unique to the current process
     */
    unsigned getThreadId(double taskStartTime, double taskEndTime);

    void setThreadIdOffset(unsigned offset);
    std::vector<unsigned> getThreadIds() const;

private:
    unsigned _startThreadId;
    std::vector<unsigned> _threadId;
};

}  // namespace profiling
}  // namespace vpux
