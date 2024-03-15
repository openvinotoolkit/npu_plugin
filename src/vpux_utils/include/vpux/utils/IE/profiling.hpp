//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ie_common.h>

#include <openvino/runtime/profiling_info.hpp>
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/profiling.hpp"
#include "vpux/utils/plugin/profiling_json.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace vpux {
namespace profiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

class TaskList;

struct TaskStatistics {
    uint64_t totalDuration;  // wall time duration of inference [ns]
    uint64_t idleDuration;   // sum all no-operation intervals [ns]
    uint64_t allTasksUnion;  // self-consistency: should match totalDuration - idleDuration [ns]

    uint64_t dmaDuration;  // sum of interval durations of a union of DMA tasks [ns].
                           // Union of tasks (coalesced tasks @see TaskTrack::coalesce() and @see
                           // TaskTrack::calculateOverlap(const TaskTrack& refTrack)) disregard possible overlaps of
                           // individual contributing tasks
    uint64_t dpuDuration;  // sum of interval durations of a union of DPU tasks [ns]
    uint64_t swDuration;   // sum of interval durations of a union of SW tasks [ns]

    uint64_t dmaDpuOverlapDuration;  // sum of interval durations of intersection of a union of DMA and a union of DPU
                                     // tasks [ns]
    uint64_t dmaSwOverlapDuration;   // sum of interval durations of intersection of a union of DMA and a union of SW
                                     // tasks [ns]
    uint64_t swDpuOverlapDuration;   // sum of interval durations of intersection of a union of SW and a union of DPU
                                     // tasks [ns]

    uint64_t dmaDpuIdleDuration;  // sum of idle (no operation) interval durations outside of a union of DMA and DPU
                                  // tasks [ns]
    uint64_t dmaSwIdleDuration;   // sum of idle interval durations outside of a union of DMA and SW tasks [ns]
    uint64_t swDpuIdleDuration;   // sum of idle interval durations outside of a union of SW and DPU tasks [ns]

    uint64_t sumOfDmaTaskDurations;  // sum of all DMA tasks durations [ns].
    uint64_t sumOfDpuTaskDurations;  // sum of all DPU tasks durations [ns]
    uint64_t sumOfSwTaskDurations;   // sum of all SW tasks durations [ns]

    /**
     * @brief calculate joint duration of SW tasks union that does not intersect with union of DPU tasks
     *
     * @return joint intervals duration in ns
     */
    int64_t getSwDurationWithoutDpuOverlap() const {
        return swDuration - swDpuOverlapDuration;
    }

    /**
     * @brief calculate joint duration of DMA tasks union that does not intersect with union of all other tasks
     *
     * @return joint intervals duration in ns
     */
    int64_t getDmaDurationWithoutOverlap() const {
        return dmaDuration - dmaSwOverlapDuration - dmaDpuOverlapDuration + swDpuOverlapDuration;
    }
};

enum class OutputType { TEXT, JSON, DEBUG };

enum class ProfilingFormat { NONE, JSON, TEXT, RAW };

// This function decodes profiling buffer into readable format.
// Format can be either regular text or TraceEvent json.
// outputFile is an optional argument - path to a file to store output. stdout if empty.
void outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                  const std::pair<const uint8_t*, size_t>& profiling, const std::string& outputFile,
                  VerbosityLevel verbosity, bool fpga, bool highFreqPerfClk);

/**
 * @brief calculate tasks timing statistics
 *
 * @param taskProfiling - vector of tasks used for calculation
 * @return task timing statistics
 */
TaskStatistics calculateTasksStatistics(const TaskList& taskProfiling);

void printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                const std::vector<LayerInfo>& layerProfiling, std::ostream& out_stream,
                                Logger log = Logger::global().nest("TraceEventExporter", 1));

void printProfilingAsText(const std::vector<TaskInfo>& taskProfiling, const std::vector<LayerInfo>& layerProfiling,
                          std::ostream& out_stream);

void printDebugProfilingInfo(const RawData& rawProfData, std::ostream& out_stream);

bool isClusterLevelProfilingTask(const TaskInfo& task);

std::ofstream openProfilingStream(ProfilingFormat* format);

void saveProfilingDataToFile(ProfilingFormat format, std::ostream& outfile,
                             const std::vector<LayerInfo>& layerProfiling, const std::vector<TaskInfo>& taskProfiling);

void saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& outfile);

LayerStatistics convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo);

template <TaskInfo::ExecType T>
bool isTask(const TaskInfo& task) {
    return task.exec_type == T;
}

/**
 * @brief Profiling tasks selection and management utility
 */
class TaskList : public std::vector<TaskInfo> {
public:
    TaskList();
    TaskList(const std::vector<TaskInfo>& tasks);

    TaskList& append(const TaskList&);
    TaskList selectClusterLevelTasks() const;
    TaskList selectTasksFromCluster(unsigned clusterId) const;
    TaskList selectDMAtasks() const;
    TaskList selectDPUtasks() const;
    TaskList selectUPAtasks() const;
    TaskList selectSWtasks() const;
    TaskList getSortedByStartTime() const;
    /**
     * @brief Infer the Cluster Count from tasks names
     *
     * @return number of different clusters the tasks are assigned to.
     *
     * The returned value may be 0 if tasks do not contain CLUSTER_LEVEL_PROFILING_SUFFIX
     * in their name.
     */
    unsigned getClusterCount() const;

    /**
     * @brief Calculate sum of all tasks durations
     *
     * @return sum of all tasks durations
     */
    int getSumOfDurations() const;

    /**
     * @brief Walltime duration of all tasks in units defined by TaskInfo
     * @return time elapsed from the first chronologically task start time
     * to the last chronologically task end time
     *
     * Note: Tasks do not need to be ordered chronologically.
     */
    int getTotalDuration() const;

private:
    /**
     * @brief Get first chronologically task start time
     *
     * @return minimal start time among all tasks
     */
    int getStartTime() const;

    /**
     * @brief Get last chronologically task end time
     *
     * @return maximal end time among all tasks
     */
    int getEndTime() const;

    void sortByStartTime();
    template <TaskInfo::ExecType T>
    TaskList selectTasksOfType() const;
};

class TraceEventExporter : private std::vector<TraceEventDesc> {
public:
    explicit TraceEventExporter(std::ostream& outStream, Logger& log);
    TraceEventExporter(const TraceEventExporter&) = delete;
    TraceEventExporter& operator=(const TraceEventExporter&) = delete;

    /**
     * @brief flush queued trace events to output stream.
     *
     */
    void flushAsTraceEvents();

    void logTasksStatistics();

    void processTasks(const std::vector<TaskInfo>& tasks);
    void processLayers(const std::vector<LayerInfo>& layers);

private:
    void createTraceEventHeader();
    void createTraceEventFooter();

    /**
     * @brief helper function to ease exporting profiled tasks to JSON format
     *
     * @param tasks - list of tasks to be exported
     * @param processName - trace event process name
     * @param createNewProcess - if true, meta data about the current trace event process and threads assigned to the
     * tasks being processed are exported as meta trace events
     *
     * The function schedules tasks for output to out stream and generates meta type header trace events.
     * It internally manages trace events' thread IDs and names.
     */
    void processTraceEvents(const TaskList& tasks, const std::string& processName, bool createNewProcess = true);

    /**
     * @brief set tracing event process name for given process id.
     *
     * @param suffixStr - suffix added at the end of line. Except for the last tracing event, JSON events are separated
     * by commas
     * @param processId - trace event process identifier
     * @param suffixStr - end of line string for the process name trace event
     */
    void setTraceEventProcessName(const std::string& processName, unsigned processId,
                                  const std::string& suffixStr = ",");

    void setTraceEventThreadName(const std::string& threadName, unsigned threadId, unsigned processId,
                                 const std::string& suffixStr = ",");

    /**
     * @brief Set the Tracing Event Process Sort Index
     *
     * @param processId - trace event process identifier
     * @param sortIndex - index defining the process ordering in the output report. (Some UIs do not respect this value)
     * @param suffixStr - suffix added at the end of line. Except for the last tracing event, JSON events are separated
     * by
     */
    void setTraceEventProcessSortIndex(unsigned processId, unsigned sortIndex, const std::string& suffixStr = ",");

    /**
     * @brief export tasks statistics to JSON
     *
     * The block containing the statistics will be called "taskStatistics"
     */
    void exportTasksStatistics() const;

    /**
     * @brief Perform basic sanity checks on task name and duration
     *
     * @param task - task to check
     *
     * For reporting tasks it is assumed that all tasks should have
     * task name format compliant with:
     *
     *  originalLayerName?t_layerType/suffix1_value1/suffix2_value2...
     *
     * This method checks for existence of suffix separator (?) in task name and
     * asserts cluster_id suffix exists for the relevant task types.
     *
     * Warning is issued if task duration is not a positive integer.
     */
    void validateTaskNameAndDuration(const TaskInfo& task) const;

    std::ostream& _outStream;
    Logger _log;
    unsigned _processId = -1;
    unsigned _threadId = 0;
    TaskStatistics _taskStatistics{};
};

/**
 * @brief TrackEvent is used to store information about count of parallel tasks at any given time during execution flow.
 */
struct TrackEvent {
    uint64_t time;      // track event time
    int taskCount = 0;  // number of tasks at given time
    bool isStart;       // indicates whether the event marks the start time of a task
    uint64_t duration;  // duration of task the event is associated with
};

/**
 * @brief TaskTrack is an NPU tasks container that stores tasks start and end times as single time events called
 * TrackEvent It encapsulates coalescing overlapping events
 *
 */
class TaskTrack {
public:
    TaskTrack& append(const TaskTrack& taskTrack);
    TaskTrack& insert(const TaskList& tasks);
    TaskTrack& insert(const TrackEvent& v);
    TaskTrack& insert(uint64_t trackTime, uint64_t evtDuration, bool isEvtStart);
    TaskTrack& sortByTime();

    /**
     * @brief calculate number of stacked tasks as a function of event time
     *
     * @return std::map<int, int> containing time and task count
     */
    std::map<int, int> getTrackProfile();

    /**
     * @brief calculate union of all events in the track.
     * The coalesced version of the sequence of events has overlapping events merged together and output
     * as a single longer event. Neighboring events are not merged. Zero-duration events are reduced out.
     */
    TaskTrack& coalesce();

    std::vector<TrackEvent> getEvents() const;
    std::vector<TrackEvent>& getEvents();

    /**
     * @return uint64_t - sum of durations of all events in the track
     */
    uint64_t getSumOfDurations() const;

    /**
     * @brief calculate tasks mutual overlap and idle times
     *
     * @param refTrack - track to calculate overlap with
     * @return pair of overlapTime and idleTime
     *
     * The overlapTime and idleTime are defined as follows:
     *
     *      test track (20 time units long): xxxx......xxxxx.....
     * reference track (20 time units long): ...yyy.......yyyyyyy
     *                             analysis:    o  iiii   oo
     *
     * o - overlap time
     * i - idle time
     *
     * totalDuration = 20
     * workload = N(x) + N(y) = 9 + 10 = 19
     * idleTime = N(i) = 4
     * overlapTime = N(o) = 3
     *
     * Note:
     *
     * The impact of zero-duration tasks is that they affect the "total duration"
     * time (i.e. inference time) but do not contribute to workload,
     * hence if duration-zero task is located at the beginning or at the end
     * of the provided task list, this may impact the statistics
     * by increasing the total execution time (totalDuration). This situation
     * is not shown in the example above.
     *
     * Default unit is ns.
     *
     */
    std::pair<uint64_t, uint64_t> calculateOverlap(const TaskTrack& refTrack) const;

private:
    std::vector<TrackEvent> _trackEvents;
};

}  // namespace profiling
}  // namespace vpux
