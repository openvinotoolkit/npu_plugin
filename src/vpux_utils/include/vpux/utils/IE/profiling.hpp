//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ie_common.h>

#include "vpux/utils/core/logger.hpp"
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

using LayerStatistics = std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>;

enum class OutputType { NONE, TEXT, JSON, DEBUG };

enum class ProfilingFormat { NONE, JSON, TEXT, RAW };

// This function decodes profiling buffer into readable format.
// Format can be either regular text or TraceEvent json.
// outputFile is an optional argument - path to a file to store output. stdout if empty.
void outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                  const std::pair<const uint8_t*, size_t>& profiling, const std::string& outputFile,
                  VerbosityLevel verbosity, bool fpga);

void printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                const std::vector<LayerInfo>& layerProfiling, std::ostream& out_stream);

void printProfilingAsText(const std::vector<TaskInfo>& taskProfiling, const std::vector<LayerInfo>& layerProfiling,
                          std::ostream& out_stream);

void printDebugProfilingInfo(const RawData& rawProfData, std::ostream& out_stream);

void printSummary(const SummaryInfo& summary, std::ostream& out_stream);

bool isClusterLevelProfilingTask(const TaskInfo& task);

bool isLowLevelProfilingTask(const TaskInfo& task);

std::ofstream openProfilingStream(ProfilingFormat* format);

void saveProfilingDataToFile(ProfilingFormat format, std::ostream& outfile,
                             const std::vector<LayerInfo>& layerProfiling, const std::vector<TaskInfo>& taskProfiling);

void saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& outfile);

LayerStatistics convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo);

template <typename T>
bool profilingTaskStartTimeComparator(const T& a, const T& b) {
    const auto namesCompareResult = std::strcmp(a.name, b.name);
    return std::forward_as_tuple(a.start_time_ns, namesCompareResult, b.duration_ns) <
           std::forward_as_tuple(b.start_time_ns, 0, a.duration_ns);
}

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
    TaskList selectTopLevelTasks() const;
    TaskList selectClusterTasks() const;
    TaskList selectClusterLevelTasks() const;
    TaskList selectClusterLevelTasks(unsigned clusterId) const;
    TaskList selectLowLevelTasks() const;
    TaskList selectDMAtasks() const;
    TaskList selectDPUtasks() const;
    TaskList selectUPAtasks() const;
    TaskList selectSWtasks() const;
    TaskList getSortedByStartTime() const;
    unsigned getClusterCount() const;

private:
    Logger _log;
    void sortByStartTime();
    template <TaskInfo::ExecType T>
    TaskList selectTasksOfType() const;
};

class TraceEventExporter : private std::vector<TraceEventDesc> {
public:
    explicit TraceEventExporter(std::ostream& outStream);
    TraceEventExporter(const TraceEventExporter&) = delete;
    TraceEventExporter& operator=(const TraceEventExporter&) = delete;
    ~TraceEventExporter();

    /**
     * @brief flush queued trace events to output stream.
     *
     */
    void flushAsTraceEvents();

    void processTasks(const std::vector<TaskInfo>& tasks);
    void processLayers(const std::vector<LayerInfo>& layers);

private:
    void createTraceEventHeader();
    void createTraceEventFooter();

    /**
     * @brief helper function to ease exporting profiled tasks to JSON format
     *
     * @param tasks - list of tasks to be exported
     * @param processId - process id to assign to the exported trace events
     * Tasks from consecutive clusters will be placed in processes with larger id values.
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
     */
    void setTraceEventProcessName(const std::string& processName, unsigned processId,
                                  const std::string& suffixStr = ",");

    void setTraceEventThreadName(const std::string& threadName, unsigned threadId, unsigned processId,
                                 const std::string& suffixStr = ",");

    /**
     * @brief Set the Tracing Event Process Sort Index
     *
     * @param sortIndex - index defining the process ordering in the output report. (Some UIs do not respect this value)
     * @param suffixStr - suffix added at the end of line. Except for the last tracing event, JSON events are separated
     * by
     */
    void setTraceEventProcessSortIndex(unsigned processId, unsigned sortIndex, const std::string& suffixStr = ",");

    std::ostream& _outStream;
    Logger _log;
    std::ios::fmtflags _origFlags;
    unsigned _processId;
    unsigned _threadId;
};

}  // namespace profiling
}  // namespace vpux
