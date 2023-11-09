//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/profiling.hpp"

#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/plugin/profiling_json.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace vpux;
using namespace vpux::profiling;
namespace ie = InferenceEngine;

namespace {

const std::string CLUSTER_PROCESS_NAME = "Cluster";
const std::string DMA_PROCESS_NAME = "DMA";
const std::string LAYER_PROCESS_NAME = "Layers";
const std::string UPA_PROCESS_NAME = "UPA";

const std::string VARIANT_NAME = "Variants";
const std::string SHAVE_NAME = "Shave";
const std::string LAYER_THREAD_NAME = "Layers";

const std::string DMA_TASK_CATEGORY = "DMA";
const std::string DPU_TASK_CATEGORY = "DPU";
const std::string NONE_TASK_CATEGORY = "NONE";
const std::string SW_TASK_CATEGORY = "SW";
const std::string UPA_TASK_CATEGORY = "UPA";

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {
        {TaskInfo::ExecType::NONE, NONE_TASK_CATEGORY}, {TaskInfo::ExecType::DPU, DPU_TASK_CATEGORY},
        {TaskInfo::ExecType::SW, SW_TASK_CATEGORY},     {TaskInfo::ExecType::DMA, DMA_TASK_CATEGORY},
        {TaskInfo::ExecType::UPA, UPA_TASK_CATEGORY},
};

/**
 * @brief Extract a value from a structured task name string
 *
 * @param name - structured task name string in format prefix1/prefix2/key1_val1/key2_val2
 * @param key - keyword to have value extracted eg: "/key1_"
 * @return std::string - extracted value starting a character after '_' and ending on either the end of the string
 * or a keyword delimiter '/'
 *
 * Eg.
 *
 * For "prefix/key1_val1/key2_val2" and key "key1_", the function yields "val1",
 * for "prefix/key1_val1/key2_val2" and key "key2_", the function yields "val2",
 * for "prefix/key1_val1/key2_val2" and key "key3_", the function yields ""
 *
 */
std::string getValueFromStructuredTaskName(const std::string& name, std::string key) {
    auto keyIdx = name.rfind(key);
    if (keyIdx == std::string::npos) {
        return "";
    }
    const auto suffix = name.substr(keyIdx);
    const auto valueBeginPos = suffix.find("_") + 1;
    const auto valueEndPos = suffix.rfind("/");

    if (valueEndPos == std::string::npos) {
        auto s = suffix.substr(valueBeginPos);
        return suffix.substr(valueBeginPos);
    }

    const auto valueLen = valueEndPos - valueBeginPos;
    return suffix.substr(valueBeginPos, valueLen);
}

std::string getClusterFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, CLUSTER_LEVEL_PROFILING_SUFFIX);
}

std::string getVariantFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, VARIANT_LEVEL_PROFILING_SUFFIX);
}

std::string getTileFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, TILE_LEVEL_PROFILING_SUFFIX);
}

std::string createClusterThreadName(const std::string& taskName, bool isSwCluster, bool lowLevel = true) {
    const std::string highLevelName = isSwCluster ? SW_TASK_CATEGORY : DPU_TASK_CATEGORY;
    const std::string clusterId = getClusterFromName(taskName);
    const std::string lowLevelId = isSwCluster ? getTileFromName(taskName) : getVariantFromName(taskName);
    if (lowLevelId.empty() || !lowLevel) {
        return highLevelName;
    }

    const std::string taskSpecifier = isSwCluster ? SHAVE_NAME : VARIANT_NAME;
    return highLevelName + " / " + taskSpecifier;
}

bool isVariantLevelProfilingTask(const TaskInfo& task) {
    bool hasVariantInName = strstr(task.name, VARIANT_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return hasVariantInName;
}

bool isTileLevelProfilingTask(const TaskInfo& task) {
    bool hasTileInName = strstr(task.name, TILE_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return hasTileInName;
}

void printDebugProfilingInfoForOneCategory(const RawProfilingRecords& records, std::ostream& outStream,
                                           size_t commonOffset) {
    using DebugFormattableRecordPtr = std::shared_ptr<DebugFormattableRecordMixin>;
    std::map<size_t, DebugFormattableRecordPtr> orderedRecords;
    for (const auto& record : records) {
        const auto debugRecordPtr = std::dynamic_pointer_cast<DebugFormattableRecordMixin>(record);
        orderedRecords[debugRecordPtr->getInMemoryOffset()] = debugRecordPtr;
    }

    bool firstTime = true;
    for (const auto& offsetAndRecord : orderedRecords) {
        const auto record = offsetAndRecord.second;
        const auto taskOffset = offsetAndRecord.first * record->getDebugDataSize();
        if (firstTime) {
            record->printDebugHeader(outStream);
            firstTime = false;
        }
        const auto taskGlobalOffset = commonOffset + taskOffset;
        const auto asRawRecord = std::dynamic_pointer_cast<RawProfilingRecord>(record);
        outStream << std::setw(14) << std::hex << taskGlobalOffset << std::setw(15) << taskOffset << std::setw(14)
                  << asRawRecord->getRecordTypeName() << std::setw(100) << asRawRecord->getTaskName();
        record->printDebugInfo(outStream);
        outStream << std::endl;
    }
}

void printDebugWorkpointsSetup(const RawProfilingData& rawProfData, std::ostream& outStream) {
    if (!rawProfData.hasWorkpointConfig()) {
        return;
    }
    const auto workpointDbgInfos = rawProfData.workpointsConfiguration;

    outStream << std::setw(14) << "Global offset" << std::setw(25) << "Engine" << std::setw(17) << "PLL Value"
              << std::setw(15) << "WRKPNT CFGID" << std::endl;
    for (const auto& workpointDbgInfo : workpointDbgInfos) {
        const auto workpointCfg = workpointDbgInfo.first;
        const auto offset = workpointDbgInfo.second;

        outStream << std::setw(14) << offset << std::setw(25) << "WORKPOINT" << std::setw(17)
                  << workpointCfg.pllMultiplier << std::setw(15) << workpointCfg.configId << std::endl;
    }
}

};  // namespace

bool vpux::profiling::isClusterLevelProfilingTask(const TaskInfo& task) {
    bool hasClusterInName = strstr(task.name, CLUSTER_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return hasClusterInName && !isLowLevelProfilingTask(task);
}

bool vpux::profiling::isLowLevelProfilingTask(const TaskInfo& task) {
    return isVariantLevelProfilingTask(task) || isTileLevelProfilingTask(task);
}

static void streamWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                         const std::pair<const uint8_t*, size_t>& profiling, std::ostream& output,
                         VerbosityLevel verbosity, bool fpga) {
    const auto blobData = blob.first;
    const auto blobSize = blob.second;
    const auto profilingData = profiling.first;
    const auto profilingSize = profiling.second;

    if (profilingType == OutputType::DEBUG) {
        const auto rawProfData = getRawProfilingTasks(blobData, blobSize, profilingData, profilingSize, TaskType::ALL,
                                                      /*ignoreSanitizationErrors =*/true);
        const auto summary = getSummary(rawProfData, profilingSize);
        printDebugProfilingInfo(rawProfData, output);
        printSummary(summary, output);
        return;
    }
    std::vector<TaskInfo> taskProfiling =
            getTaskInfo(blobData, blobSize, profilingData, profilingSize, TaskType::ALL, verbosity, fpga);
    std::vector<LayerInfo> layerProfiling = getLayerInfo(taskProfiling);

    // Order tasks and layers by start time
    std::sort(taskProfiling.begin(), taskProfiling.end(), profilingTaskStartTimeComparator<TaskInfo>);
    std::sort(layerProfiling.begin(), layerProfiling.end(), profilingTaskStartTimeComparator<LayerInfo>);

    switch (profilingType) {
    case OutputType::TEXT:
        printProfilingAsText(taskProfiling, layerProfiling, output);
        break;
    case OutputType::JSON:
        printProfilingAsTraceEvent(taskProfiling, layerProfiling, output);
        break;
    // case OutputType::DEBUG:
    case OutputType::NONE:
        break;
    default:
        std::cerr << "Unsupported profiling output type." << std::endl;
        break;
    }
};

void vpux::profiling::printProfilingAsText(const std::vector<TaskInfo>& taskProfiling,
                                           const std::vector<LayerInfo>& layerProfiling, std::ostream& outStream) {
    uint64_t last_time_ns = 0;
    std::ios::fmtflags origFlags(outStream.flags());
    outStream << std::left << std::setprecision(2) << std::fixed;
    for (auto& task : taskProfiling) {
        std::string exec_type_str;
        std::string taskName(task.name);

        switch (task.exec_type) {
        case TaskInfo::ExecType::DMA:
            exec_type_str = "DMA";
            outStream << "Task(" << exec_type_str << "): " << std::setw(60) << taskName
                      << "\tTime(us): " << std::setw(8) << (float)task.duration_ns / 1000
                      << "\tStart(us): " << std::setw(8) << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::DPU:
            exec_type_str = "DPU";
            outStream << "Task(" << exec_type_str << "): " << std::setw(60) << taskName
                      << "\tTime(us): " << std::setw(8) << (float)task.duration_ns / 1000
                      << "\tStart(us): " << std::setw(8) << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::SW:
            exec_type_str = "SW";
            outStream << "Task(" << exec_type_str << "): " << std::setw(60) << taskName
                      << "\tTime(us): " << std::setw(8) << (float)task.duration_ns / 1000
                      << "\tCycles:" << task.active_cycles << "(" << task.stall_cycles << ")"
                      << "\tStart(us): " << std::setw(8) << (float)task.start_time_ns / 1000 << std::endl;
            break;
        case TaskInfo::ExecType::UPA:
            exec_type_str = "UPA";
            outStream << "Task(" << exec_type_str << "): " << std::setw(60) << taskName
                      << "\tTime(us): " << std::setw(8) << (float)task.duration_ns / 1000
                      << "\tCycles:" << task.active_cycles << "(" << task.stall_cycles << ")"
                      << "\tStart(us): " << std::setw(8) << (float)task.start_time_ns / 1000 << std::endl;
            break;
        default:
            break;
        }

        uint64_t task_end_time_ns = task.start_time_ns + task.duration_ns;
        if (last_time_ns < task_end_time_ns) {
            last_time_ns = task_end_time_ns;
        }
    }

    uint64_t total_time = 0;
    for (auto& layer : layerProfiling) {
        outStream << "Layer: " << std::setw(40) << layer.name << " Type: " << std::setw(20) << layer.layer_type
                  << " DPU: " << std::setw(8) << (float)layer.dpu_ns / 1000 << " SW: " << std::setw(8)
                  << (float)layer.sw_ns / 1000 << " DMA: " << std::setw(8) << (float)layer.dma_ns / 1000
                  << "\tStart: " << (float)layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    outStream << "Total time: " << (float)total_time / 1000 << "us, Real: " << (float)last_time_ns / 1000 << "us"
              << std::endl;
    outStream.flags(origFlags);
}

std::string getTraceEventThreadName(const TaskInfo& task, bool lowLevel = true) {
    if (task.exec_type == TaskInfo::ExecType::DMA) {
        return DMA_TASK_CATEGORY;
    } else if (task.exec_type == TaskInfo::ExecType::UPA) {
        return SW_TASK_CATEGORY;  // we use SW task labels for UPA threads
    }
    bool isSwExecType = task.exec_type == TaskInfo::ExecType::SW;
    return createClusterThreadName(task.name, isSwExecType, lowLevel);
}

void vpux::profiling::printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                                 const std::vector<LayerInfo>& layerProfiling,
                                                 std::ostream& outStream) {
    TraceEventExporter events(outStream);
    events.processTasks(taskProfiling);
    events.processLayers(layerProfiling);
    events.flushAsTraceEvents();
}

void vpux::profiling::printDebugProfilingInfo(const RawData& profData, std::ostream& outStream) {
    const auto rawProfData = profData.rawRecords;
    for (const auto& typeAndOffset : rawProfData.parseOrder) {
        const auto tasks = rawProfData.getTaskOfType(typeAndOffset.first);
        printDebugProfilingInfoForOneCategory(tasks, outStream, typeAndOffset.second);
    }
    printDebugWorkpointsSetup(rawProfData, outStream);
}

void vpux::profiling::printSummary(const SummaryInfo& summary, std::ostream& outStream) {
    outStream << std::endl
              << std::setw(6) << "Engine" << std::setw(15) << "Entry size" << std::setw(20) << "Number of tasks"
              << std::setw(15) << "Offset" << std::setw(15) << "Buffer size" << std::endl;
    auto printSectionInfo = [](const SummaryInfo::SectionInfo& si, std::string engine, std::ostream& outStream) {
        outStream << std::setw(6) << engine << std::setw(15) << si.entrySize << std::setw(20) << si.numOfTasks
                  << std::setw(15) << si.bufferOffset << std::setw(15) << si.bufferSize << std::endl;
    };
    printSectionInfo(summary.dpuInfo, "DPU", outStream);
    printSectionInfo(summary.swInfo, "SW", outStream);
    printSectionInfo(summary.dmaInfo, "DMA", outStream);

    outStream << "Expected profiling buffer size = " << summary.totalBufferSize << std::endl;
}

void vpux::profiling::outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                                   const std::pair<const uint8_t*, size_t>& profiling, const std::string& filename,
                                   VerbosityLevel verbosity, bool fpga) {
    if (filename.empty()) {
        streamWriter(profilingType, blob, profiling, std::cout, verbosity, fpga);
    } else {
        std::ofstream outfile;
        outfile.open(filename, std::ios::out | std::ios::trunc);
        if (outfile.is_open()) {
            streamWriter(profilingType, blob, profiling, outfile, verbosity, fpga);
            outfile.close();
        } else {
            std::cerr << "Can't write result into " << filename << std::endl;
        }
    }
}

std::string getEnvVar(const std::string& varName, bool capitalize, const std::string& defaultValue = "") {
    const char* rawValue = std::getenv(varName.c_str());
    std::string value = rawValue == nullptr ? defaultValue : rawValue;

    if (capitalize) {
        std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    }
    return value;
}

ProfilingFormat getProfilingFormat(const std::string& format) {
    if (format == "JSON")
        return ProfilingFormat::JSON;
    if (format == "TEXT")
        return ProfilingFormat::TEXT;
    if (format == "RAW")
        return ProfilingFormat::RAW;
    return ProfilingFormat::NONE;
}

std::ofstream vpux::profiling::openProfilingStream(ProfilingFormat* format) {
    const auto printProfiling = getEnvVar("NPU_PRINT_PROFILING", true);
    IE_ASSERT(format != nullptr);
    *format = getProfilingFormat(printProfiling);

    std::ofstream outFile;
    if (*format != ProfilingFormat::NONE) {
        const auto outFileName = getEnvVar("NPU_PROFILING_OUTPUT_FILE", false);
        auto flags = std::ios::out | std::ios::trunc;
        if (*format == ProfilingFormat::RAW) {
            flags |= std::ios::binary;
        }
        outFile.open(outFileName, flags);
        if (!outFile) {
            VPUX_THROW("Can't write into file {1}", outFileName);
        }
    }
    return outFile;
}

void vpux::profiling::saveProfilingDataToFile(ProfilingFormat format, std::ostream& outfile,
                                              const std::vector<LayerInfo>& layerProfiling,
                                              const std::vector<TaskInfo>& taskProfiling) {
    const auto DEFAULT_VERBOSE_VALUE = "HIGH";
    static const std::map<std::string, size_t> VERBOSITY_TO_NUM_FILTERS = {
            {"LOW", 2},
            {"MEDIUM", 1},
            {"HIGH", 0},
    };
    auto verbosityValue = getEnvVar("NPU_PROFILING_VERBOSITY", true, DEFAULT_VERBOSE_VALUE);
    if (VERBOSITY_TO_NUM_FILTERS.count(verbosityValue) == 0) {
        verbosityValue = DEFAULT_VERBOSE_VALUE;
    }
    std::vector<decltype(&isLowLevelProfilingTask)> verbosityFilters = {&isLowLevelProfilingTask,
                                                                        &isClusterLevelProfilingTask};
    std::vector<TaskInfo> filteredTasks;
    // Driver return tasks at maximum verbosity, so filter them to needed level
    std::copy_if(taskProfiling.begin(), taskProfiling.end(), std::back_inserter(filteredTasks),
                 [&](const TaskInfo& task) {
                     bool toKeep = true;
                     for (size_t filterId = 0; filterId < VERBOSITY_TO_NUM_FILTERS.at(verbosityValue); ++filterId) {
                         toKeep &= !verbosityFilters[filterId](task);
                     }
                     return toKeep;
                 });
    switch (format) {
    case ProfilingFormat::JSON:
        printProfilingAsTraceEvent(filteredTasks, layerProfiling, outfile);
        break;
    case ProfilingFormat::TEXT:
        printProfilingAsText(filteredTasks, layerProfiling, outfile);
        break;
    case ProfilingFormat::RAW:
    case ProfilingFormat::NONE:
        IE_ASSERT(false);
    }
}

void vpux::profiling::saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& outfile) {
    outfile.write(reinterpret_cast<const char*>(rawBuffer), size);
    outfile.flush();
}

LayerStatistics vpux::profiling::convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo) {
    LayerStatistics perfCounts;
    int execution_index = 0;
    for (const auto& layer : layerInfo) {
        ie::InferenceEngineProfileInfo info;
        info.status = ie::InferenceEngineProfileInfo::EXECUTED;
        info.realTime_uSec = layer.duration_ns / 1000;
        info.execution_index = execution_index++;
        auto execLen = sizeof(info.exec_type);
        if (layer.sw_ns > 0) {
            strncpy(info.exec_type, "SW", execLen);
        } else if (layer.dpu_ns > 0) {
            strncpy(info.exec_type, "DPU", execLen);
        } else {
            strncpy(info.exec_type, "DMA", execLen);
        }
        info.exec_type[execLen - 1] = 0;
        info.cpu_uSec = (layer.dma_ns + layer.sw_ns + layer.dpu_ns) / 1000;
        auto typeLen = sizeof(info.layer_type);
        strncpy(info.layer_type, layer.layer_type, typeLen);
        info.layer_type[typeLen - 1] = 0;
        perfCounts[layer.name] = info;
    }
    return perfCounts;
}

vpux::profiling::TaskList::TaskList(): _log(Logger::global().nest("TaskList", 1)) {
}

vpux::profiling::TaskList::TaskList(const std::vector<TaskInfo>& tasks)
        : std::vector<TaskInfo>(tasks), _log(Logger::global().nest("TaskList", 1)) {
}

TaskList vpux::profiling::TaskList::selectTopLevelTasks() const {
    std::vector<TaskInfo> selectedTasks;
    for (const auto& task : *this) {
        if (!isClusterLevelProfilingTask(task) && !isLowLevelProfilingTask(task)) {
            selectedTasks.push_back(task);
        }
    }
    return TaskList(selectedTasks);
}

template <TaskInfo::ExecType T>
TaskList vpux::profiling::TaskList::selectTasksOfType() const {
    std::vector<TaskInfo> selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isTask<T>);
    return TaskList(selectedTasks);
}

TaskList vpux::profiling::TaskList::selectDPUtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DPU>();
}

TaskList vpux::profiling::TaskList::selectUPAtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::UPA>();
}

TaskList vpux::profiling::TaskList::selectDMAtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DMA>();
}

TaskList vpux::profiling::TaskList::selectSWtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::SW>();
}

TaskList vpux::profiling::TaskList::getSortedByStartTime() const {
    TaskList sorted(*this);
    sorted.sortByStartTime();
    return sorted;
}

TaskList vpux::profiling::TaskList::selectLowLevelTasks() const {
    std::vector<TaskInfo> selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isLowLevelProfilingTask);
    return TaskList(selectedTasks);
}

TaskList vpux::profiling::TaskList::selectClusterTasks() const {
    std::vector<TaskInfo> selectedTasks;
    for (const auto& task : *this) {
        if ((task.exec_type == TaskInfo::ExecType::SW && isLowLevelProfilingTask(task)) ||
            task.exec_type == TaskInfo::ExecType::DPU) {
            selectedTasks.push_back(task);
        }
    }
    return TaskList(selectedTasks);
}

TaskList vpux::profiling::TaskList::selectClusterLevelTasks() const {
    std::vector<TaskInfo> selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isClusterLevelProfilingTask);
    return TaskList(selectedTasks);
}

TaskList vpux::profiling::TaskList::selectClusterLevelTasks(unsigned clusterId) const {
    std::vector<TaskInfo> selectedTasks;

    for (const auto& task : *this) {
        if (isClusterLevelProfilingTask(task) || isLowLevelProfilingTask(task)) {
            std::string idStr = getClusterFromName(task.name);
            unsigned id;
            try {
                size_t idx;
                id = std::stoi(idStr, &idx);
                if (idx < idStr.size()) {  // Not all characters converted, ignoring
                    _log.warning("Not all characters converted while extracting cluster id from task ({0}). Task will "
                                 "not be reported.",
                                 task.name);

                    continue;
                }
            } catch (...) {  // Could not extract cluster id
                _log.warning("Could not extract cluster id for task ({0}). Task will not be reported.", task.name);
                continue;
            }

            if (id == clusterId) {
                selectedTasks.push_back(task);
            }
        }
    }
    return TaskList(selectedTasks);
}

void vpux::profiling::TaskList::sortByStartTime() {
    std::sort(begin(), end(), profilingTaskStartTimeComparator<TaskInfo>);
}

unsigned vpux::profiling::TaskList::getClusterCount() const {
    std::set<std::string> clusterLevelThreadNames;
    for (const auto& task : *this) {
        clusterLevelThreadNames.insert(getClusterFromName(task.name));
    }
    return clusterLevelThreadNames.size();
}

void vpux::profiling::TraceEventExporter::processTasks(const std::vector<TaskInfo>& tasks) {
    //
    // Export DMA tasks
    //
    auto dmaTasks = TaskList(tasks).selectDMAtasks();
    processTraceEvents(dmaTasks, DMA_PROCESS_NAME, /* createNewProcess= */ true);

    //
    // Export cluster tasks (DPU and SW)
    //
    _threadId = 0;
    unsigned clusterCount = TaskList(tasks).selectClusterLevelTasks().getClusterCount();

    TaskList dpuTasks = TaskList(tasks).selectDPUtasks();
    bool isSingleTile =
            clusterCount == 0 && !dpuTasks.empty();  // single tile compilation that doesn't generate cluster
                                                     // id information in tasks names is handled separately

    TaskList swTasks = TaskList(tasks).selectSWtasks();
    TaskList reportedSWtasks = swTasks.selectLowLevelTasks();

    if (reportedSWtasks.empty()) {
        reportedSWtasks = swTasks.selectClusterLevelTasks();  // if there are no low-level tasks, report higher level
                                                              // tasks instead
        if (reportedSWtasks.size() != swTasks.size() && !reportedSWtasks.empty()) {
            _log.warning(
                    "Number of cluster-level SW tasks ({0}) differs from the number of all SW tasks ({1}). Some tasks "
                    "may not be exported.",
                    reportedSWtasks.size(), swTasks.size());
        }
    }

    if (isSingleTile) {
        processTraceEvents(dpuTasks, CLUSTER_PROCESS_NAME, /* createNewProcess= */ true);
        reportedSWtasks = reportedSWtasks.empty()
                                  ? swTasks
                                  : reportedSWtasks;  // case for compiler trace report with enabled HW profiling
        processTraceEvents(reportedSWtasks, CLUSTER_PROCESS_NAME, /* createNewProcess= */ dpuTasks.empty());
    } else {
        for (unsigned clusterId = 0; clusterId < clusterCount; clusterId++) {
            _threadId = 0;
            std::string processName = CLUSTER_PROCESS_NAME + " (" + std::to_string(clusterId) + ")";
            auto clusterDpuTasks = dpuTasks.selectClusterLevelTasks(clusterId);
            auto clusterSwTasks = reportedSWtasks.selectClusterLevelTasks(clusterId);
            processTraceEvents(clusterDpuTasks, processName, /* createNewProcess= */ true);
            processTraceEvents(clusterSwTasks, processName, /* createNewProcess= */ clusterDpuTasks.empty());
        }
    }

    //
    // Export non-clustered SW tasks into separate UPA process
    //
    TaskList upaTasks = TaskList(tasks).selectUPAtasks();
    processTraceEvents(upaTasks, UPA_PROCESS_NAME, /* createNewProcess= */ true);
}

void vpux::profiling::TraceEventExporter::processLayers(const std::vector<LayerInfo>& layers) {
    //
    // Export layers
    //

    if (layers.empty()) {
        return;
    }

    TraceEventTimeOrderedDistribution layersDistr;
    _processId++;

    for (auto& layer : layers) {
        struct TraceEventDesc ted;
        ted.name = layer.name;
        ted.category = "Layer";
        ted.pid = _processId;
        ted.timestamp = layer.start_time_ns / 1000.;
        ted.duration = layer.duration_ns / 1000.;

        // provide ns-resolution integers to avoid round-off errors during fixed precision output to JSON
        ted.tid = layersDistr.getThreadId(layer.start_time_ns, layer.start_time_ns + layer.duration_ns);
        ted.customArgs.push_back({"Layer type", layer.layer_type});
        push_back(ted);
    }

    setTraceEventProcessName(LAYER_PROCESS_NAME, _processId);
    setTraceEventProcessSortIndex(_processId, _processId);
    for (unsigned layer_id : layersDistr.getThreadIds()) {
        setTraceEventThreadName(LAYER_THREAD_NAME, layer_id, _processId);
    }
}

void vpux::profiling::TraceEventExporter::processTraceEvents(const TaskList& tasks, const std::string& processName,
                                                             bool createNewProcess) {
    if (tasks.empty()) {  // don't need to output process details if there are no tasks to export
        return;
    }
    if (createNewProcess) {
        _processId++;
        _threadId = 0;
    } else {
        _threadId++;
    }

    int maxThreadId = _threadId;
    TraceEventTimeOrderedDistribution threadDistr(_threadId);
    auto sortedTasks = tasks.getSortedByStartTime();
    std::map<unsigned, std::string> thName;

    for (const auto& task : sortedTasks) {
        struct TraceEventDesc ted;
        ted.name = task.name;
        ted.category = enumToStr.at(task.exec_type);
        ted.timestamp = task.start_time_ns / 1000.;
        ted.duration = task.duration_ns / 1000.;

        // provide ns-resolution integers to avoid round-off errors during fixed precision output to JSON
        ted.tid = threadDistr.getThreadId(task.start_time_ns, task.start_time_ns + task.duration_ns);
        ted.pid = _processId;
        thName[ted.tid] = getTraceEventThreadName(task);
        maxThreadId = std::max(ted.tid, maxThreadId);

        push_back(ted);
    }

    if (createNewProcess) {
        // Set process name and sort index
        setTraceEventProcessName(processName, _processId);
        setTraceEventProcessSortIndex(_processId, _processId);
    }

    // update thread name
    for (unsigned idOffset = 0; idOffset < thName.size(); idOffset++) {
        unsigned id = _threadId + idOffset;
        setTraceEventThreadName(thName[id], id, _processId);
    }

    _threadId = maxThreadId;
}

vpux::profiling::TraceEventExporter::TraceEventExporter(std::ostream& outStream)
        : _outStream(outStream),
          _log(Logger::global().nest("TraceEventExporter", 1)),
          _origFlags(outStream.flags()),
          _processId(-1),
          _threadId(0) {
    createTraceEventHeader();
}

vpux::profiling::TraceEventExporter::~TraceEventExporter() {
    createTraceEventFooter();
    _outStream.flags(_origFlags);
}

void vpux::profiling::TraceEventExporter::flushAsTraceEvents() {
    for (auto tedIt = this->begin(); tedIt != std::prev(this->end()); ++tedIt) {
        _outStream << *tedIt << "," << std::endl;
    }
    _outStream << back() << std::endl;
}

void vpux::profiling::TraceEventExporter::createTraceEventHeader() {
    // Trace Events timestamps are in microseconds, set precision to preserve nanosecond resolution
    _outStream << std::setprecision(3) << "{\"traceEvents\":[" << std::endl;
}

void vpux::profiling::TraceEventExporter::createTraceEventFooter() {
    _outStream << "],\n"
               // Hint for a classic Perfetto UI to use nanoseconds for display
               // JSON timestamps are expected to be in microseconds regardless
               << "\"displayTimeUnit\": \"ns\"\n"
               << "}" << std::endl;
}

void vpux::profiling::TraceEventExporter::setTraceEventProcessName(const std::string& processName, unsigned processId,
                                                                   const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "process_name", "ph": "M", "pid":)") << processId
               << R"(, "args": {"name" : ")" << processName << R"("}})" << suffixStr << std::endl;
}

void vpux::profiling::TraceEventExporter::setTraceEventThreadName(const std::string& threadName, unsigned threadId,
                                                                  unsigned processId, const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "thread_name", "ph": "M", "pid":)") << processId << R"(, "tid":)" << threadId
               << R"(, "args": {"name" : ")" << threadName << R"("}})" << suffixStr << std::endl;
}

void vpux::profiling::TraceEventExporter::setTraceEventProcessSortIndex(unsigned processId, unsigned sortIndex,
                                                                        const std::string& suffixStr) {
    _outStream << std::string(R"({"name": "process_sort_index", "ph": "M", "pid":)") << processId
               << R"(, "args": {"sort_index" : ")" << sortIndex << R"("}})" << suffixStr << std::endl;
}
