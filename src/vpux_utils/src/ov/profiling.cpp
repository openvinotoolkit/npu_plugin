//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <openvino/core/except.hpp>

#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/plugin/profiling_json.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

using namespace vpux;
using namespace vpux::profiling;

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

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {{TaskInfo::ExecType::NONE, NONE_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::DPU, DPU_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::SW, SW_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::DMA, DMA_TASK_CATEGORY},
                                                                    {TaskInfo::ExecType::UPA, UPA_TASK_CATEGORY}

};

bool isVariantLevelProfilingTask(const TaskInfo& task) {
    const std::string variantSuffix = LOCATION_SEPARATOR + VARIANT_LEVEL_PROFILING_SUFFIX + "_";
    bool hasVariantInName = getTaskNameSuffixes(task.name).find(variantSuffix) != std::string::npos;

    return hasVariantInName;
}

void printDebugProfilingInfoSection(const RawProfilingRecords& records, std::ostream& outStream, size_t commonOffset) {
    using DebugFormattableRecordPtr = std::shared_ptr<DebugFormattableRecordMixin>;
    std::map<size_t, DebugFormattableRecordPtr> orderedRecords;
    for (const auto& record : records) {
        const auto debugRecordPtr = std::dynamic_pointer_cast<DebugFormattableRecordMixin>(record);
        VPUX_THROW_WHEN(debugRecordPtr == nullptr, "Expected formatable record");
        orderedRecords[debugRecordPtr->getInMemoryOffset()] = debugRecordPtr;
    }

    bool firstTime = true;
    const auto ostreamFlags = outStream.flags();
    for (const auto& offsetAndRecordIdx : orderedRecords | indexed) {
        const auto index = offsetAndRecordIdx.index();
        const auto offsetAndRecord = offsetAndRecordIdx.value();
        const auto record = offsetAndRecord.second;
        const auto taskOffset = offsetAndRecord.first * record->getDebugDataSize();
        if (firstTime) {
            outStream << std::setw(8) << "Index" << std::setw(8) << "Offset" << std::setw(14) << "Engine";
            record->printDebugHeader(outStream);
            outStream << std::left << std::setw(2) << ""
                      << "Task" << std::right << '\n';
            firstTime = false;
        }
        const auto taskGlobalOffset = commonOffset + taskOffset;
        const auto asRawRecord = std::dynamic_pointer_cast<RawProfilingRecord>(record);
        VPUX_THROW_WHEN(asRawRecord == nullptr, "Invalid record");
        outStream << std::setw(8) << std::dec << index << std::setw(8) << std::hex << taskGlobalOffset << std::setw(14)
                  << convertExecTypeToName(asRawRecord->getExecutorType());
        record->printDebugInfo(outStream);
        outStream << std::left << std::setw(2) << "" << asRawRecord->getTaskName() << std::right << '\n';
    }
    outStream << std::endl;
    outStream.flags(ostreamFlags);
}

void printDebugWorkpointsSetup(const RawProfilingData& rawProfData, std::ostream& outStream) {
    const auto workpointDbgInfos = rawProfData.workpoints;
    if (workpointDbgInfos.empty()) {
        return;
    }

    const auto ostreamFlags = outStream.flags();
    outStream << std::hex << std::setw(8) << "Index" << std::setw(8) << "Offset" << std::setw(14) << "Engine"
              << std::setw(17) << "PLL Value" << std::setw(15) << "CFGID" << std::endl;
    for (const auto& workpointDbgInfoIdx : workpointDbgInfos | indexed) {
        const auto index = workpointDbgInfoIdx.index();
        const auto workpointDbgInfo = workpointDbgInfoIdx.value();
        const auto workpointCfg = workpointDbgInfo.first;
        const auto offset = workpointDbgInfo.second;

        outStream << std::hex << std::setw(8) << index << std::setw(8) << offset << std::setw(14)
                  << convertExecTypeToName(ExecutorType::WORKPOINT) << std::setw(17) << workpointCfg.pllMultiplier
                  << std::setw(15) << workpointCfg.configId << std::endl;
    }
    outStream.flags(ostreamFlags);
}

};  // namespace

bool vpux::profiling::isClusterLevelProfilingTask(const TaskInfo& task) {
    const std::string clusterSuffix = LOCATION_SEPARATOR + CLUSTER_LEVEL_PROFILING_SUFFIX + "_";
    bool hasClusterInName = getTaskNameSuffixes(task.name).find(clusterSuffix) != std::string::npos;
    return hasClusterInName && !isVariantLevelProfilingTask(task);
}

static void streamWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                         const std::pair<const uint8_t*, size_t>& profiling, std::ostream& output,
                         VerbosityLevel verbosity, bool fpga, bool /*highFreqPerfClk*/) {
    const auto blobData = blob.first;
    const auto blobSize = blob.second;
    const auto profilingData = profiling.first;
    const auto profilingSize = profiling.second;

    if (profilingType == OutputType::DEBUG) {
        const auto rawProfData = getRawProfilingTasks(blobData, blobSize, profilingData, profilingSize,
                                                      /*ignoreSanitizationErrors =*/true);
        printDebugProfilingInfo(rawProfData, output);
        return;
    }
    std::vector<TaskInfo> taskProfiling = getTaskInfo(blobData, blobSize, profilingData, profilingSize, verbosity, fpga,
                                                      /* ignoreSanitizationErrors */ false);
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
    default:
        VPUX_THROW("Unsupported profiling output type.");
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

std::string getTraceEventThreadName(const TaskInfo& task) {
    switch (task.exec_type) {
    case TaskInfo::ExecType::DMA:
        return DMA_TASK_CATEGORY;
    case TaskInfo::ExecType::UPA:
        return SW_TASK_CATEGORY;  // we use SW task labels for UPA threads
    case TaskInfo::ExecType::SW:
        return SW_TASK_CATEGORY + " / " + SHAVE_NAME;
    case TaskInfo::ExecType::DPU:
        return isVariantLevelProfilingTask(task) ? DPU_TASK_CATEGORY + " / " + VARIANT_NAME : DPU_TASK_CATEGORY;
    default:
        VPUX_THROW("Unknown task category");
    }
}

TaskStatistics vpux::profiling::calculateTasksStatistics(const TaskList& tasks) {
    TaskStatistics stats = TaskStatistics();
    auto dmaStatsTasks = tasks.selectDMAtasks();
    auto dpuStatsTasks = tasks.selectDPUtasks().selectClusterLevelTasks();  // for task statistics use invariants only
    auto swTasks = tasks.selectSWtasks();
    auto upaTasks = tasks.selectUPAtasks();
    VPUX_THROW_WHEN(!swTasks.empty() && !upaTasks.empty(),
                    "UPA and Shave tasks should be mutually exclusive but are found to coexist");

    TaskList swStatsTasks;
    if (swTasks.empty()) {
        swStatsTasks.append(upaTasks);  // for task statistics use all UPA tasks
    } else {
        swStatsTasks.append(swTasks);  // we use all SW tasks considered as low-level tasks being directly profiled
                                       // and assuming no SW task grouping is performed
    }

    stats.totalDuration = static_cast<uint64_t>(tasks.getTotalDuration());

    // DMA stats
    TaskTrack dmaTrack;
    stats.dmaDuration = dmaTrack.insert(dmaStatsTasks).coalesce().getSumOfDurations();
    stats.sumOfDmaTaskDurations = dmaStatsTasks.getSumOfDurations();

    // DPU stats
    TaskTrack dpuTrack;
    stats.dpuDuration = dpuTrack.insert(dpuStatsTasks).coalesce().getSumOfDurations();
    stats.sumOfDpuTaskDurations = dpuStatsTasks.getSumOfDurations();

    // SW (UPA and Shave) stats
    TaskTrack swTrack;
    stats.swDuration = swTrack.insert(swStatsTasks).coalesce().getSumOfDurations();
    stats.sumOfSwTaskDurations = swStatsTasks.getSumOfDurations();

    // DMA vs DPU overlap statistics
    auto overlapIdleDurations = dmaTrack.calculateOverlap(dpuTrack);
    stats.dmaDpuOverlapDuration = overlapIdleDurations.first;
    stats.dmaDpuIdleDuration = overlapIdleDurations.second;

    // DMA vs SW overlap statistics
    overlapIdleDurations = dmaTrack.calculateOverlap(swTrack);
    stats.dmaSwOverlapDuration = overlapIdleDurations.first;
    stats.dmaSwIdleDuration = overlapIdleDurations.second;

    // SW vs DPU overlap statistics
    overlapIdleDurations = swTrack.calculateOverlap(dpuTrack);
    stats.swDpuOverlapDuration = overlapIdleDurations.first;
    stats.swDpuIdleDuration = overlapIdleDurations.second;

    // calculate idle time and tasks union
    TaskTrack statsTasks;
    statsTasks.insert(dmaStatsTasks);
    statsTasks.insert(dpuStatsTasks);
    statsTasks.insert(swStatsTasks);
    statsTasks.coalesce();

    overlapIdleDurations = statsTasks.calculateOverlap(statsTasks);
    stats.allTasksUnion = overlapIdleDurations.first;  // set intersection with self is union
    stats.idleDuration = stats.totalDuration - stats.allTasksUnion;

    return stats;
}

void vpux::profiling::TraceEventExporter::logTasksStatistics() {
    _log.info("Tasks statistics:");
    auto log = _log.nest();
    auto& stats = _taskStatistics;

    log.info("- total duration [ns]: {0}", stats.totalDuration);
    log.info("- DMA duration [ns]: {0} ({1} %)", stats.dmaDuration,
             double(stats.dmaDuration) / stats.totalDuration * 100);
    log.info("- DPU duration [ns]: {0} ({1} %)", stats.dpuDuration,
             double(stats.dpuDuration) / stats.totalDuration * 100);
    log.info("- SW duration [ns]: {0} ({1} %)", stats.swDuration, double(stats.swDuration) / stats.totalDuration * 100);

    // tasks overlap statistics
    log.info("- DMA-DPU overlap [ns]: {0} ({1} %)", stats.dmaDpuOverlapDuration,
             double(stats.dmaDpuOverlapDuration) / stats.totalDuration * 100);
    log.info("- DMA-SW overlap [ns]: {0} ({1} %)", stats.dmaSwOverlapDuration,
             double(stats.dmaSwOverlapDuration) / stats.totalDuration * 100);
    log.info("- SW-DPU overlap [ns]: {0} ({1} %)", stats.swDpuOverlapDuration,
             double(stats.swDpuOverlapDuration) / stats.totalDuration * 100);
    log.info("- all tasks union [ns]: {0} ({1} %)", stats.allTasksUnion,
             double(stats.allTasksUnion) / stats.totalDuration * 100);

    // tasks idle statistics
    log.info("- total idle [ns]: {0} ({1} %)", stats.idleDuration,
             double(stats.idleDuration) / stats.totalDuration * 100);

    // SW duration that does not overlap with DPU
    auto SWdurWithoutDPUoverlap = stats.getSwDurationWithoutDpuOverlap();
    log.info("- SW duration without DPU overlap [ns]: {0} ({1} %)", SWdurWithoutDPUoverlap,
             double(SWdurWithoutDPUoverlap) / stats.totalDuration * 100);

    // DMA duration that does not overlap with SW and DPU
    auto DMAdurWithoutOverlap = stats.getDmaDurationWithoutOverlap();
    log.info("- DMA duration without overlaps [ns]: {0} ({1} %)", DMAdurWithoutOverlap,
             double(DMAdurWithoutOverlap) / stats.totalDuration * 100);

    // tiling and scheduling performance parameters
    log.info("- Sum of DMA task durations [ns]: {0} ({1} %)", stats.sumOfDmaTaskDurations,
             double(stats.sumOfDmaTaskDurations) / stats.totalDuration * 100);
    log.info("- Sum of DPU task durations [ns]: {0} ({1} %)", stats.sumOfDpuTaskDurations,
             double(stats.sumOfDpuTaskDurations) / stats.totalDuration * 100);
    log.info("- Sum of SW task durations [ns]: {0} ({1} %)", stats.sumOfSwTaskDurations,
             double(stats.sumOfSwTaskDurations) / stats.totalDuration * 100);
}

void vpux::profiling::printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                                 const std::vector<LayerInfo>& layerProfiling, std::ostream& outStream,
                                                 Logger log) {
    TraceEventExporter events(outStream, log);
    events.processTasks(taskProfiling);
    events.processLayers(layerProfiling);
    events.flushAsTraceEvents();

    events.logTasksStatistics();
}

void vpux::profiling::printDebugProfilingInfo(const RawData& profData, std::ostream& outStream) {
    const auto rawProfData = profData.rawRecords;
    for (const auto& typeAndOffset : rawProfData.parseOrder) {
        const auto tasks = rawProfData.getTaskOfType(typeAndOffset.first);
        printDebugProfilingInfoSection(tasks, outStream, typeAndOffset.second);
    }
    printDebugWorkpointsSetup(rawProfData, outStream);
}

void vpux::profiling::outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                                   const std::pair<const uint8_t*, size_t>& profiling, const std::string& filename,
                                   VerbosityLevel verbosity, bool fpga, bool highFreqPerfClk) {
    if (filename.empty()) {
        streamWriter(profilingType, blob, profiling, std::cout, verbosity, fpga, highFreqPerfClk);
    } else {
        std::ofstream outfile;
        outfile.open(filename, std::ios::out | std::ios::trunc);
        VPUX_THROW_WHEN(!outfile, "Cannot write to '{0}'", filename);
        streamWriter(profilingType, blob, profiling, outfile, verbosity, fpga, highFreqPerfClk);
    }
}

std::string getEnvVar(const char* varName, bool capitalize, const std::string& defaultValue = "") {
    const char* rawValue = std::getenv(varName);
    std::string value = (rawValue == nullptr) ? defaultValue : rawValue;

    if (capitalize) {
        std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    }
    return value;
}

std::string getProfilingFileName(ProfilingFormat format) {
    const char* name = std::getenv("NPU_PROFILING_OUTPUT_FILE");
    if (name != nullptr)
        return name;
    switch (format) {
    case ProfilingFormat::JSON:
        return "profiling.json";
    case ProfilingFormat::TEXT:
        return "profiling.txt";
    default:
        return "profiling.out";
    }
}

ProfilingFormat getProfilingFormat(const std::string& format) {
    if (format == "JSON")
        return ProfilingFormat::JSON;
    if (format == "TEXT")
        return ProfilingFormat::TEXT;
    if (format == "RAW")
        return ProfilingFormat::RAW;

    VPUX_THROW("Invalid profiling format '{0}'", format);
}

std::ofstream vpux::profiling::openProfilingStream(ProfilingFormat* format) {
    OPENVINO_ASSERT(format != nullptr);
    *format = ProfilingFormat::NONE;
    const auto printProfiling = getEnvVar("NPU_PRINT_PROFILING", true);
    if (printProfiling != "") {
        *format = getProfilingFormat(printProfiling);
    }

    std::ofstream outFile;
    if (*format != ProfilingFormat::NONE) {
        const auto outFileName = getProfilingFileName(*format);
        auto flags = std::ios::out | std::ios::trunc;
        if (*format == ProfilingFormat::RAW) {
            flags |= std::ios::binary;
        }
        outFile.open(outFileName, flags);
        if (!outFile) {
            VPUX_THROW("Can't write into file '{0}'", outFileName);
        }
    }
    return outFile;
}

void vpux::profiling::saveProfilingDataToFile(ProfilingFormat format, std::ostream& outfile,
                                              const std::vector<LayerInfo>& layerProfiling,
                                              const std::vector<TaskInfo>& taskProfiling) {
    const auto DEFAULT_VERBOSE_VALUE = "HIGH";
    static const std::map<std::string, size_t> VERBOSITY_TO_NUM_FILTERS = {
            {"LOW", 1},
            {"MEDIUM", 0},
            {"HIGH", 0},
    };
    auto verbosityValue = getEnvVar("NPU_PROFILING_VERBOSITY", true, DEFAULT_VERBOSE_VALUE);
    if (VERBOSITY_TO_NUM_FILTERS.count(verbosityValue) == 0) {
        verbosityValue = DEFAULT_VERBOSE_VALUE;
    }
    std::vector<decltype(&isVariantLevelProfilingTask)> verbosityFilters = {&isVariantLevelProfilingTask,
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
        OPENVINO_ASSERT(false);
    }
}

void vpux::profiling::saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& outfile) {
    outfile.write(reinterpret_cast<const char*>(rawBuffer), size);
    outfile.flush();
}

LayerStatistics vpux::profiling::convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo) {
    LayerStatistics perfCounts;

    for (const auto& layer : layerInfo) {
        ov::ProfilingInfo info;
        info.node_name = layer.name;
        info.status = ov::ProfilingInfo::Status::EXECUTED;
        const auto& real_time_ns = std::chrono::nanoseconds(layer.duration_ns);
        info.real_time = std::chrono::duration_cast<std::chrono::microseconds>(real_time_ns);

        if (layer.sw_ns > 0) {
            info.exec_type = "SW";
        } else if (layer.dpu_ns > 0) {
            info.exec_type = "DPU";
        } else {
            info.exec_type = "DMA";
        }

        const auto& cpu_time_ns = std::chrono::nanoseconds(layer.dma_ns + layer.sw_ns + layer.dpu_ns);
        info.cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_time_ns);
        info.node_type = layer.layer_type;
        perfCounts.push_back(info);
    }

    return perfCounts;
}

vpux::profiling::TaskList::TaskList() {
}

vpux::profiling::TaskList::TaskList(const std::vector<TaskInfo>& tasks): std::vector<TaskInfo>(tasks) {
}

template <TaskInfo::ExecType T>
TaskList vpux::profiling::TaskList::selectTasksOfType() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isTask<T>);
    return selectedTasks;
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

TaskList vpux::profiling::TaskList::selectClusterLevelTasks() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isClusterLevelProfilingTask);
    return selectedTasks;
}

TaskList vpux::profiling::TaskList::selectTasksFromCluster(unsigned clusterId) const {
    auto log = Logger::global();
    TaskList selectedTasks;

    for (const auto& task : *this) {
        std::string idStr = getClusterFromName(task.name);
        unsigned id;
        try {
            size_t idx;
            id = std::stoi(idStr, &idx);
            if (idx < idStr.size()) {  // Not all characters converted, ignoring
                log.warning("Not all characters converted while extracting cluster id from task ({0}). Task will "
                            "not be reported.",
                            task.name);
                continue;
            }
        } catch (...) {  // Could not extract cluster id
            log.warning("Could not extract cluster id for task ({0}). Task will not be reported.", task.name);
            continue;
        }

        if (id == clusterId) {
            selectedTasks.push_back(task);
        }
    }
    return selectedTasks;
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

int vpux::profiling::TaskList::getSumOfDurations() const {
    return std::accumulate(begin(), end(), 0, [](const int& totalTime, const TaskInfo& task) {
        return totalTime + task.duration_ns;
    });
}

int vpux::profiling::TaskList::getStartTime() const {
    VPUX_THROW_WHEN(empty(), "Minimal time in empty TaskList is not defined.");

    auto minElementIt = min_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return a.start_time_ns < b.start_time_ns;
    });
    return minElementIt->start_time_ns;
}

int vpux::profiling::TaskList::getEndTime() const {
    VPUX_THROW_WHEN(empty(), "Maximal time in empty TaskList is not defined.");

    auto maxElementIt = max_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return a.start_time_ns + a.duration_ns < b.start_time_ns + b.duration_ns;
    });
    return maxElementIt->start_time_ns + maxElementIt->duration_ns;
}

int vpux::profiling::TaskList::getTotalDuration() const {
    if (empty()) {
        return 0;
    }
    return getEndTime() - getStartTime();
}

TaskList& vpux::profiling::TaskList::append(const TaskList& tasks) {
    insert(end(), tasks.begin(), tasks.end());
    return *this;
}

void vpux::profiling::TraceEventExporter::processTasks(const std::vector<TaskInfo>& tasks) {
    for (auto& task : tasks) {
        validateTaskNameAndDuration(task);
    }

    //
    // Export DMA tasks
    //
    auto dmaTasks = TaskList(tasks).selectDMAtasks();
    processTraceEvents(dmaTasks, DMA_PROCESS_NAME, /* createNewProcess= */ true);

    //
    // Export cluster tasks (DPU and SW)
    //
    _threadId = 0;
    unsigned clusterCount = TaskList(tasks).getClusterCount();

    TaskList dpuTasks = TaskList(tasks).selectDPUtasks();
    TaskList swTasks = TaskList(tasks).selectSWtasks();

    for (unsigned clusterId = 0; clusterId < clusterCount; clusterId++) {
        _threadId = 0;
        std::string processName = CLUSTER_PROCESS_NAME + " (" + std::to_string(clusterId) + ")";
        auto clusterDpuTasks = dpuTasks.selectTasksFromCluster(clusterId);
        auto clusterSwTasks = swTasks.selectTasksFromCluster(clusterId);
        processTraceEvents(clusterDpuTasks, processName, /* createNewProcess= */ true);
        processTraceEvents(clusterSwTasks, processName, /* createNewProcess= */ clusterDpuTasks.empty());
    }

    //
    // Export non-clustered SW tasks into separate UPA process
    //
    TaskList upaTasks = TaskList(tasks).selectUPAtasks();
    processTraceEvents(upaTasks, UPA_PROCESS_NAME, /* createNewProcess= */ true);

    VPUX_THROW_WHEN(!upaTasks.empty() && !swTasks.empty(),
                    "UPA and Shave tasks should be mutually exclusive but are found to coexist");

    _taskStatistics = calculateTasksStatistics(tasks);
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

void vpux::profiling::TraceEventExporter::exportTasksStatistics() const {
    _outStream << std::fixed << "\"taskStatistics\": {\n"
               << "\"total duration\":" << _taskStatistics.totalDuration * 1e-3 << ",\n"
               << "\"DMA duration\":" << _taskStatistics.dmaDuration * 1e-3 << ",\n"
               << "\"DPU duration\":" << _taskStatistics.dpuDuration * 1e-3 << ",\n"
               << "\"SW duration\":" << _taskStatistics.swDuration * 1e-3 << ",\n"
               << "\"DMA-DPU overlap\":" << _taskStatistics.dmaDpuOverlapDuration * 1e-3 << ",\n"
               << "\"DMA-SW overlap\":" << _taskStatistics.dmaSwOverlapDuration * 1e-3 << ",\n"
               << "\"SW-DPU overlap\":" << _taskStatistics.swDpuOverlapDuration * 1e-3 << ",\n"
               << "\"all tasks union\":" << _taskStatistics.allTasksUnion * 1e-3 << ",\n"
               << "\"total idle\":" << _taskStatistics.idleDuration * 1e-3 << ",\n"
               << "\"SW duration without DPU overlap\":" << _taskStatistics.getSwDurationWithoutDpuOverlap() * 1e-3
               << ",\n"
               << "\"DMA duration without overlaps\":" << _taskStatistics.getDmaDurationWithoutOverlap() * 1e-3 << ",\n"
               << "\"Sum of DMA task durations\":" << _taskStatistics.sumOfDmaTaskDurations * 1e-3 << ",\n"
               << "\"Sum of DPU task durations\":" << _taskStatistics.sumOfDpuTaskDurations * 1e-3 << ",\n"
               << "\"Sum of SW task durations\":" << _taskStatistics.sumOfSwTaskDurations * 1e-3 << "\n"
               << "},\n";
}

void vpux::profiling::TraceEventExporter::validateTaskNameAndDuration(const TaskInfo& task) const {
    // check existence of cluster_id suffix in clustered tasks
    if (isTask<TaskInfo::ExecType::SW>(task) || isTask<TaskInfo::ExecType::DPU>(task)) {
        bool hasClusterInName = !getClusterFromName(task.name).empty();
        VPUX_THROW_UNLESS(hasClusterInName, "Task {0} does not have assigned cluster_id", task.name);
    }

    // check task duration
    if (task.duration_ns <= 0) {
        _log.warning("Task {0} has duration {1} ns.", task.name, task.duration_ns);
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

vpux::profiling::TraceEventExporter::TraceEventExporter(std::ostream& outStream, Logger& log)
        : _outStream(outStream), _log(log) {
    createTraceEventHeader();
}

void vpux::profiling::TraceEventExporter::flushAsTraceEvents() {
    if (!empty()) {
        for (auto tedIt = begin(); tedIt != std::prev(end()); ++tedIt) {
            _outStream << *tedIt << "," << std::endl;
        }
        _outStream << back() << std::endl;
    }
    clear();
    createTraceEventFooter();
    _outStream.flush();
}

void vpux::profiling::TraceEventExporter::createTraceEventHeader() {
    // Trace Events timestamps are in microseconds, set precision to preserve nanosecond resolution
    _outStream << std::setprecision(3) << "{\"traceEvents\":[" << std::endl;
}

void vpux::profiling::TraceEventExporter::createTraceEventFooter() {
    // close traceEvents block
    _outStream << "],\n";
    exportTasksStatistics();
    // Hint for a classic Perfetto UI to use nanoseconds for display
    // JSON timestamps are expected to be in microseconds regardless
    _outStream << "\"displayTimeUnit\": \"ns\"\n"
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

TaskTrack& vpux::profiling::TaskTrack::append(const TaskTrack& taskTrack) {
    const auto& events = taskTrack.getEvents();
    _trackEvents.insert(_trackEvents.end(), events.begin(), events.end());
    return *this;
}

TaskTrack& vpux::profiling::TaskTrack::insert(const TaskList& tasks) {
    for (const auto& task : tasks) {
        TrackEvent eventStart = {task.start_time_ns, 0, true, task.duration_ns};
        TrackEvent eventEnd = {task.start_time_ns + task.duration_ns, 0, false, task.duration_ns};
        _trackEvents.push_back(eventStart);
        _trackEvents.push_back(eventEnd);
    }
    return *this;
}

TaskTrack& vpux::profiling::TaskTrack::insert(const TrackEvent& v) {
    _trackEvents.push_back(v);
    return *this;
}

TaskTrack& vpux::profiling::TaskTrack::insert(uint64_t trackTime, uint64_t evtDuration, bool isEvtStart) {
    TrackEvent evt = {trackTime, 0, isEvtStart, evtDuration};
    _trackEvents.push_back(evt);
    return *this;
}

TaskTrack& vpux::profiling::TaskTrack::sortByTime() {
    std::sort(_trackEvents.begin(), _trackEvents.end(), [](const TrackEvent& a, const TrackEvent& b) {
        return std::make_tuple(a.time, b.duration) < std::make_tuple(b.time, a.duration);
    });
    return *this;
}

std::map<int, int> vpux::profiling::TaskTrack::getTrackProfile() {
    std::map<int, int> profile;

    int concurrencyCounter = 0;
    for (auto& evt : getEvents()) {
        if (evt.duration == 0) {  // zero-duration events do not change track profile so do not store their times
            continue;
        }

        evt.isStart ? concurrencyCounter++ : concurrencyCounter--;
        profile[evt.time] = concurrencyCounter;
    }
    return profile;
}

TaskTrack& vpux::profiling::TaskTrack::coalesce() {
    // sort events by increasing time
    sortByTime();

    const int noTaskValue = 0;  // value indicating that no tasks are present at given time
    int startTime = 0;
    bool findNewTaskStart = true;
    auto trackProfile = getTrackProfile();
    std::vector<TrackEvent> coalescedEvents;
    for (auto it = trackProfile.begin(); it != trackProfile.end(); ++it) {
        int currentTime = it->first;
        int stackedTasksCount = it->second;
        TrackEvent evt;

        evt.isStart = stackedTasksCount != noTaskValue;
        if (stackedTasksCount > noTaskValue && findNewTaskStart) {
            findNewTaskStart = false;
            startTime = currentTime;
        }

        if (stackedTasksCount == noTaskValue) {
            evt.duration = currentTime - startTime;
            evt.time = startTime;
            evt.taskCount = stackedTasksCount;
            coalescedEvents.push_back(evt);
            findNewTaskStart = true;
        }
    }

    _trackEvents = std::move(coalescedEvents);
    return *this;
}

std::vector<TrackEvent>& vpux::profiling::TaskTrack::getEvents() {
    return _trackEvents;
}

std::vector<TrackEvent> vpux::profiling::TaskTrack::getEvents() const {
    return _trackEvents;
}

uint64_t vpux::profiling::TaskTrack::getSumOfDurations() const {
    return std::accumulate(_trackEvents.begin(), _trackEvents.end(), 0,
                           [](const int& totalTime, const TrackEvent& trackEvent) {
                               return totalTime + trackEvent.duration;
                           });
}

std::pair<uint64_t, uint64_t> vpux::profiling::TaskTrack::calculateOverlap(const TaskTrack& refTrack) const {
    TaskTrack combined;
    combined.append(*this).append(refTrack);

    // create a list of relevant times of event overlaps
    TaskTrack trackEvents;
    for (auto& x : combined.getEvents()) {
        if (x.duration > 0) {  // ignore zero-duration events
            trackEvents.insert(x.time, x.duration, true);
            trackEvents.insert(x.time + x.duration, x.duration, false);
        }
    }

    // calculate overlap
    trackEvents.sortByTime();
    int counter = 0;
    for (auto& trackEvent : trackEvents.getEvents()) {
        trackEvent.isStart ? counter++ : counter--;
        trackEvent.taskCount = counter;
    }

    // calculate concurrent tasks and idle durations
    uint64_t overlapTime = 0;
    uint64_t idleTime = 0;
    std::vector<TrackEvent>::iterator it;
    const int concurrentTasksThres = 2;  // at least two tasks are present at given time
    const int noTaskValue = 0;           // value indicating that no tasks are present at given time
    for (it = trackEvents.getEvents().begin(); it != trackEvents.getEvents().end(); ++it) {
        auto currentTimestamp = it->time;
        auto stackedTasksCount = it->taskCount;
        auto nextTimestamp = next(it, 1) == trackEvents.getEvents().end() ? it->time : std::next(it, 1)->time;
        if (stackedTasksCount >= concurrentTasksThres) {  // at least two tasks are executed in parallel
                                                          // starting from the current event
            overlapTime += nextTimestamp - currentTimestamp;
        } else if (stackedTasksCount == noTaskValue) {  // no tasks are executed starting from current event
            idleTime += nextTimestamp - currentTimestamp;
        }
    }

    return std::make_pair(overlapTime, idleTime);
}
