//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/IE/prefix.hpp"

#include "vpux/utils/plugin/profiling_json.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

using namespace vpux;
using namespace vpux::profiling;
namespace ie = InferenceEngine;

namespace {

static constexpr int PID = 1, TID = 1;

const std::string NONE_THREAD_NAME = "NONE";
const std::string DMA_THREAD_NAME = "DMA";
const std::string SW_THREAD_NAME = "SW";
const std::string DPU_THREAD_NAME = "DPU";
const std::string DPU_CLUSTERS_THREAD_NAME = "DPU Clusters";
const std::string SW_CLUSTERS_THREAD_NAME = "SW Clusters";
const std::string LAYER_THREAD_NAME = "Layers";

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {
        {TaskInfo::ExecType::NONE, NONE_THREAD_NAME},
        {TaskInfo::ExecType::DPU, DPU_THREAD_NAME},
        {TaskInfo::ExecType::SW, SW_THREAD_NAME},
        {TaskInfo::ExecType::DMA, DMA_THREAD_NAME},
};

static const std::map<TaskInfo::ExecType, std::string> enumToClusterStr = {
        {TaskInfo::ExecType::DPU, DPU_CLUSTERS_THREAD_NAME},
        {TaskInfo::ExecType::SW, SW_CLUSTERS_THREAD_NAME},
};

std::string getClusterFromName(const std::string& name) {
    const auto suffix = name.substr(name.rfind(CLUSTER_LEVEL_PROFILING_SUFFIX));
    const auto clusterBeginPos = suffix.find("_") + 1;
    const auto clusterEndPos = suffix.rfind("/");
    // Havent variant suffix
    if (clusterEndPos == suffix.npos) {
        return suffix.substr(clusterBeginPos);
    }

    const auto clusterLen = clusterEndPos - clusterBeginPos;
    return suffix.substr(clusterBeginPos, clusterLen);
}

std::string createClusterThreadName(const std::string& clusterId, bool isSwCluster) {
    const std::string prefix = isSwCluster ? "SW " : "";
    return prefix + "Cluster[" + clusterId + "]";
}

bool isSwTask(const TaskInfo& task) {
    return task.exec_type == TaskInfo::ExecType::SW;
}

std::vector<std::string> generateThreadNames(const std::vector<TaskInfo>& tasks) {
    // isSwCluster and Cluster ID
    std::map<TaskInfo::ExecType, std::set<std::string>> numsOfClusters;
    bool hasVariantsData = false;
    for (const auto& task : tasks) {
        if (isClusterLevelProfilingTask(task) == true) {
            numsOfClusters[task.exec_type].insert(getClusterFromName(task.name));
        }
        hasVariantsData = hasVariantsData || isLowLevelProfilingTask(task);
    }
    // Mandatory threads
    std::vector<std::string> tidNames = {DMA_THREAD_NAME};
    // Clustered executors
    for (const auto execType : {TaskInfo::ExecType::DPU, TaskInfo::ExecType::SW}) {
        bool isSwExecType = execType == TaskInfo::ExecType::SW;
        for (const auto& clusterId : numsOfClusters[execType]) {
            tidNames.push_back(createClusterThreadName(clusterId, isSwExecType));
        }
        if (!numsOfClusters[execType].empty()) {
            tidNames.push_back(enumToClusterStr.at(execType));
        }
        tidNames.push_back(enumToStr.at(execType));
    }
    tidNames.push_back(LAYER_THREAD_NAME);
    return tidNames;
}

bool isVariantLevelProfilingTask(const TaskInfo& task) {
    bool hasVariantInName = strstr(task.name, VARIANT_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return task.exec_type == TaskInfo::ExecType::DPU && hasVariantInName == true;
}

bool isTileLevelProfilingTask(const TaskInfo& task) {
    bool hasTileInName = strstr(task.name, TILE_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return isSwTask(task) && hasTileInName == true;
}

};  // namespace

bool vpux::profiling::isClusterLevelProfilingTask(const TaskInfo& task) {
    bool hasClusterInName = strstr(task.name, CLUSTER_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    bool isDPUorActTask = task.exec_type == TaskInfo::ExecType::DPU || isSwTask(task);
    return isDPUorActTask && hasClusterInName == true && isLowLevelProfilingTask(task) == false;
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

    std::vector<TaskInfo> taskProfiling;
    std::vector<LayerInfo> layerProfiling;
    std::vector<DebugInfo> debugProfiling;
    SummaryInfo summary;

    if (profilingType == OutputType::DEBUG) {
        debugProfiling = getTaskInfoInDebugMode(blobData, blobSize, profilingData, profilingSize, TaskType::ALL);
        summary = getSummary(blobData, profilingSize);
    } else {
        taskProfiling = getTaskInfo(blobData, blobSize, profilingData, profilingSize, TaskType::ALL, verbosity, fpga);
        layerProfiling = getLayerInfo(taskProfiling);
    }

    switch (profilingType) {
    case OutputType::TEXT:
        printProfilingAsText(taskProfiling, layerProfiling, output);
        break;
    case OutputType::JSON:
        printProfilingAsTraceEvent(taskProfiling, layerProfiling, output);
        break;
    case OutputType::DEBUG:
        printDebugProfilingInfo(debugProfiling, output);
        printSummary(summary, output);
        break;
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
    outStream << std::left << std::setprecision(2) << std::fixed;
    for (auto& task : taskProfiling) {
        std::string exec_type_str;
        std::string taskName = std::string(task.name);

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
        outStream << "Layer: " << std::setw(40) << layer.name << " DPU: " << std::setw(8) << (float)layer.dpu_ns / 1000
                  << " SW: " << std::setw(8) << (float)layer.sw_ns / 1000 << " DMA: " << std::setw(8)
                  << (float)layer.dma_ns / 1000 << "\tStart: " << (float)layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    outStream << "Total time: " << (float)total_time / 1000 << "us, Real: " << (float)last_time_ns / 1000 << "us"
              << std::endl;
}

void vpux::profiling::printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                                 const std::vector<LayerInfo>& layerProfiling,
                                                 std::ostream& outStream) {
    struct TracingEventDesc ted;
    ted.pid = PID;
    const auto tidNames = generateThreadNames(taskProfiling);
    auto findTid = [&](auto& tidName) {
        // Adding 2 make offset from initial thread
        return static_cast<unsigned>(std::find(tidNames.begin(), tidNames.end(), tidName) - tidNames.begin() + 2);
    };

    // Trace Events timestamps are in microseconds, set precision to preserve nanosecond resolution
    outStream << std::setprecision(3) << "{\"traceEvents\":[" << std::endl;

    for (auto& task : taskProfiling) {
        ted.name = task.name;
        ted.category = enumToStr.at(task.exec_type);
        std::string tidName = ted.category;
        bool isSwExecType = isSwTask(task);
        if (isClusterLevelProfilingTask(task)) {
            tidName = isSwExecType ? SW_CLUSTERS_THREAD_NAME : DPU_CLUSTERS_THREAD_NAME;
        } else if (isLowLevelProfilingTask(task)) {
            tidName = createClusterThreadName(getClusterFromName(task.name), isSwExecType);
        }

        ted.tid = findTid(tidName);
        ted.timestamp = task.start_time_ns / 1000.;
        ted.duration = task.duration_ns / 1000.;
        outStream << ted;
    }

    ted.category = "Layer";
    for (auto& layer : layerProfiling) {
        ted.name = layer.name;
        ted.tid = findTid(LAYER_THREAD_NAME);
        ted.timestamp = layer.start_time_ns / 1000.;
        ted.duration = layer.duration_ns / 1000.;
        outStream << ted;
    }

    // Last item so far has a comma in the end. If the stream was std::cout this comma can't be removed.
    // Adding metadata items in the end to make JSON correct.
    outStream << std::string(R"({"name": "process_name", "ph": "M", "pid": )") << PID << R"(, "tid": )" << TID
              << R"(, "args": {"name" : "Inference"}},)" << std::endl;
    for (size_t tid = 0; tid < tidNames.size(); tid++) {
        outStream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )" << tid + 2
                  << R"(, "args": {"name" : ")" << tidNames[tid] << R"("}})";
        if (tid + 1 != tidNames.size()) {
            outStream << ",";
        }
        outStream << std::endl;
    }
    outStream << "],\n"
              // Hint for a classic Perfetto UI to use nanoseconds for display
              // JSON timestamps are expected to be in microseconds regardless
              << "\"displayTimeUnit\": \"ns\"\n"
              << "}" << std::endl;
}

static std::string rtToString(const RecordType rt) {
    static const std::map<RecordType, std::string> dict{
            {RecordType::DMA20, "DMA 2.0"}, {RecordType::DMA27, "DMA 2.7"}, {RecordType::DPU_HWP27, "HWP DPU"},
            {RecordType::DPU_SW, "SW DPU"}, {RecordType::SW_UPA, "UPA"},    {RecordType::SW_ACT, "ACT"},
    };
    return dict.at(rt);
}

static constexpr int COL_WIDTH_32 = 11;
static constexpr int COL_WIDTH_64 = 19;

void vpux::profiling::printDebugProfilingInfo(const std::vector<DebugInfo>& debugProfiling, std::ostream& outStream) {
    using ColDesc = std::vector<std::pair<std::string, int>>;
    uint32_t commonOffset = 0;
    auto header = [](bool& firstTime, const ColDesc& columns, std::ostream& outStream, const DebugInfo& task,
                     uint32_t commonOffset) {
        if (firstTime) {
            firstTime = false;
            outStream << std::setw(7) << "Offset" << std::setw(7) << "Offset" << std::setw(10) << "Engine"
                      << std::setw(100) << "Layer name";
            for (const std::pair<std::string, int>& p : columns) {
                outStream << std::setw(p.second) << p.first;
            }
            outStream << std::endl;
        }
        outStream << std::setw(7) << std::hex << commonOffset << std::setw(7) << task.offset << std::setw(10)
                  << rtToString(task.recordType) << std::setw(100) << task.name;
    };

    bool dmaTitles = true;
    bool dpuTitles = true;
    bool swTitles = true;
    const ColDesc dmaCol{{"Timestamp", COL_WIDTH_64}};
    const ColDesc swDpuCol{{"Begin tstamp", COL_WIDTH_64}, {"End tstamp", COL_WIDTH_64}};
    const ColDesc hwpDpuCol{{"IDU dur", COL_WIDTH_32}, {"IDU tstamp", COL_WIDTH_32}, {"SWE ID", 7}, {"Res", 4},
                            {"ODU dur", COL_WIDTH_32}, {"ODU tstamp", COL_WIDTH_32}, {"Res", 7}};
    const ColDesc upaCol{{"Begin tstamp", COL_WIDTH_64},
                         {"End tstamp", COL_WIDTH_64},
                         {"Stall", COL_WIDTH_32},
                         {"Active", COL_WIDTH_32}};
    const ColDesc actShaveCol{{"Begin", COL_WIDTH_64}, {"Duration", COL_WIDTH_32}, {"Stall", COL_WIDTH_32}};
    for (const DebugInfo& task : debugProfiling) {
        switch (task.recordType) {
        case RecordType::DMA20:
            header(dmaTitles, dmaCol, outStream, task, commonOffset);
            outStream << std::setw(dmaCol[0].second) << task.raw.dma20;
            commonOffset += 4;
            break;
        case RecordType::DMA27:
            header(dmaTitles, dmaCol, outStream, task, commonOffset);
            outStream << std::setw(dmaCol[0].second) << task.raw.dma27;
            commonOffset += 8;
            break;
        case RecordType::DPU_HWP27: {
            header(dpuTitles, hwpDpuCol, outStream, task, commonOffset);
            const HwpDpuMode0Data_t& hwpDpu = task.raw.hwpDpu;
            outStream << std::setw(hwpDpuCol[0].second) << hwpDpu.idu_wl_duration << std::setw(hwpDpuCol[1].second)
                      << hwpDpu.idu_tstamp << std::setw(hwpDpuCol[2].second) << hwpDpu.sve_id
                      << std::setw(hwpDpuCol[3].second) << hwpDpu.reserved3 << std::setw(hwpDpuCol[4].second)
                      << hwpDpu.odu_wl_duration << std::setw(hwpDpuCol[5].second) << hwpDpu.odu_tstamp
                      << std::setw(hwpDpuCol[6].second) << hwpDpu.reserved8;
            commonOffset += sizeof(HwpDpuMode0Data_t);
            break;
        }
        case RecordType::DPU_SW:
            header(dpuTitles, swDpuCol, outStream, task, commonOffset);
            outStream << std::setw(swDpuCol[0].second) << task.raw.swDpu.begin << std::setw(swDpuCol[1].second)
                      << task.raw.swDpu.end;
            commonOffset += sizeof(SwDpuData_t);
            break;
        case RecordType::SW_UPA:
            header(swTitles, upaCol, outStream, task, commonOffset);
            outStream << std::setw(upaCol[0].second) << task.raw.upa.begin << std::setw(upaCol[1].second)
                      << task.raw.upa.end << std::setw(upaCol[2].second) << task.raw.upa.stallCycles
                      << std::setw(upaCol[3].second) << task.raw.upa.activeCycles;
            commonOffset += sizeof(UpaData_t);
            break;
        case RecordType::SW_ACT:
            header(swTitles, actShaveCol, outStream, task, commonOffset);
            outStream << std::setw(actShaveCol[0].second) << task.raw.actShave.begin << std::setw(actShaveCol[1].second)
                      << task.raw.actShave.duration << std::setw(actShaveCol[2].second)
                      << task.raw.actShave.stallCycles;
            commonOffset += sizeof(ActShaveData_t);
            break;
        default:
            break;
        }
        outStream << std::endl;
    }
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
