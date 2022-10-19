//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/IE/profiling.hpp"

#include "vpux/utils/plugin/profiling_json.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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
const std::string CLUSTERS_THREAD_NAME = "Clusters";
const std::string EXEC_SUM_THREAD_NAME = "Sum of execution times";
const std::string LAYER_THREAD_NAME = "Layer execution time";

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {
        {TaskInfo::ExecType::NONE, NONE_THREAD_NAME},
        {TaskInfo::ExecType::DPU, DPU_THREAD_NAME},
        {TaskInfo::ExecType::SW, SW_THREAD_NAME},
        {TaskInfo::ExecType::DMA, DMA_THREAD_NAME},
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

std::string createClusterThreadName(const std::string& clusterId) {
    return "Cluster " + clusterId;
}

std::vector<std::string> generateThreadNames(const std::vector<TaskInfo>& tasks) {
    std::set<std::string> clusterIds;
    bool hasVariantsData = false;
    for (const auto& task : tasks) {
        if (isClusterLevelProfilingTask(task) == true) {
            clusterIds.emplace(getClusterFromName(task.name));
        }
        hasVariantsData |= isVariantLevelProfilingTask(task);
    }
    // Mandatory threads
    std::vector<std::string> tidNames = {DMA_THREAD_NAME, SW_THREAD_NAME};
    // If we have cluster level tasks insert it
    if (clusterIds.empty() == false) {
        // If we have variants insert them before cluster thread
        if (hasVariantsData == true) {
            for (const auto clusterId : clusterIds) {
                tidNames.push_back(createClusterThreadName(clusterId));
            }
        }
        tidNames.push_back(CLUSTERS_THREAD_NAME);
    }
    // Appending other mandatory threads
    for (const auto& other : {DPU_THREAD_NAME, EXEC_SUM_THREAD_NAME, LAYER_THREAD_NAME}) {
        tidNames.push_back(other);
    }
    return tidNames;
}

};  // namespace

bool vpux::profiling::isClusterLevelProfilingTask(const TaskInfo& task) {
    bool hasClusterInName = strstr(task.name, CLUSTER_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return task.exec_type == TaskInfo::ExecType::DPU && hasClusterInName == true &&
           isVariantLevelProfilingTask(task) == false;
}

bool vpux::profiling::isVariantLevelProfilingTask(const TaskInfo& task) {
    bool hasVariantInName = strstr(task.name, VARIANT_LEVEL_PROFILING_SUFFIX.c_str()) != nullptr;
    return task.exec_type == TaskInfo::ExecType::DPU && hasVariantInName == true;
}

std::map<std::string, ie::InferenceEngineProfileInfo> vpux::profiling::convertProfilingLayersToIEInfo(
        std::vector<LayerInfo>& layerInfo) {
    std::map<std::string, ie::InferenceEngineProfileInfo> perfCounts;

    int execution_index = 0;
    std::map<std::string, int> layerNames;
    for (const auto& layer : layerInfo) {
        ie::InferenceEngineProfileInfo info;
        auto name = std::string(layer.name);

        // Prevent existence of the same layer names
        auto layerNameIt = layerNames.find(name);
        if (layerNameIt != layerNames.end()) {
            layerNames[name]++;
            name += "/" + std::to_string(layerNames[name]);
        } else
            layerNames[name] = 0;

        info.status = ie::InferenceEngineProfileInfo::EXECUTED;
        info.realTime_uSec = layer.duration_ns / 1000;
        info.execution_index = execution_index++;
        auto typeLen = sizeof(info.exec_type);
        if (layer.sw_ns > 0) {
            strncpy(info.exec_type, "SW", typeLen - 1);
        } else if (layer.dpu_ns > 0) {
            strncpy(info.exec_type, "DPU", typeLen - 1);
        } else {
            strncpy(info.exec_type, "DMA", typeLen - 1);
        }
        info.exec_type[typeLen - 1] = '\0';
        info.cpu_uSec = (layer.dma_ns + layer.sw_ns + layer.dpu_ns) / 1000;
        typeLen = sizeof(info.layer_type);
        strncpy(info.layer_type, layer.layer_type, typeLen - 1);
        info.layer_type[typeLen - 1] = '\0';
        perfCounts[name] = info;
    }

    return perfCounts;
}

static void streamWriter(const OutputType profilingType, const std::pair<const uint8_t*, uint64_t>& blob,
                         const std::pair<const uint8_t*, uint64_t>& profiling, std::ostream& output,
                         TimeUnitFormat format, VerbosityLevel verbosity) {
    const auto blobData = blob.first;
    const auto blobSize = blob.second;
    const auto profilingData = profiling.first;
    const auto profilingSize = profiling.second;

    std::vector<TaskInfo> taskProfiling =
            getTaskInfo(blobData, blobSize, profilingData, profilingSize, TaskType::ALL, verbosity);
    std::vector<LayerInfo> layerProfiling = getLayerInfo(taskProfiling);

    switch (profilingType) {
    case OutputType::TEXT:
        printProfilingAsText(taskProfiling, layerProfiling, output);
        break;
    case OutputType::JSON:
        printProfilingAsTraceEvent(taskProfiling, layerProfiling, output, format);
        break;
    case OutputType::NONE:
        break;
    default:
        std::cerr << "Unsupported profiling output type." << std::endl;
        break;
    }
};

void vpux::profiling::printProfilingAsText(const std::vector<TaskInfo>& taskProfiling,
                                           const std::vector<LayerInfo>& layerProfiling, std::ostream& out_stream) {
    uint64_t last_time_ns = 0;
    for (auto& task : taskProfiling) {
        std::string exec_type_str;
        switch (task.exec_type) {
        case TaskInfo::ExecType::DMA:
            exec_type_str = "DMA";
            out_stream << "Task(" << exec_type_str << "): " << std::setw(80) << task.name
                       << "\tTime: " << (float)task.duration_ns / 1000 << "\tStart: " << task.start_time_ns / 1000
                       << std::endl;
            break;
        case TaskInfo::ExecType::DPU:
            exec_type_str = "DPU";
            out_stream << "Task(" << exec_type_str << "): " << std::setw(80) << task.name
                       << "\tTime: " << (float)task.duration_ns / 1000 << "\tStart: " << task.start_time_ns / 1000
                       << std::endl;
            break;
        case TaskInfo::ExecType::SW:
            exec_type_str = "SW";
            out_stream << "Task(" << exec_type_str << "): " << std::setw(80) << task.name
                       << "\tTime: " << (float)task.duration_ns / 1000 << "\tCycles:" << task.active_cycles << "("
                       << task.stall_cycles << ")"
                       << "\tStart: " << task.start_time_ns / 1000 << std::endl;
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
        out_stream << "Layer: " << std::setw(80) << layer.name << " DPU: " << std::setw(5) << layer.dpu_ns / 1000
                   << " SW: " << std::setw(5) << layer.sw_ns / 1000 << " DMA: " << std::setw(5) << layer.dma_ns / 1000
                   << "\tStart: " << layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    out_stream << "TotalTime: " << total_time / 1000 << "us, Real: " << last_time_ns / 1000 << "us" << std::endl;
}

void vpux::profiling::printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                                 const std::vector<LayerInfo>& layerProfiling, std::ostream& out_stream,
                                                 TimeUnitFormat format) {
    struct TracingEventDesc ted;
    ted.pid = PID;
    const auto tidNames = generateThreadNames(taskProfiling);
    auto findTid = [&](auto& tidName) {
        // Adding 2 make offset from initial thread
        return static_cast<unsigned>(std::find(tidNames.begin(), tidNames.end(), tidName) - tidNames.begin() + 2);
    };

    out_stream << "{\"traceEvents\":[" << format << std::endl;

    for (auto& task : taskProfiling) {
        ted.name = task.name;
        ted.category = enumToStr.at(task.exec_type);
        std::string tidName = ted.category;
        if (isClusterLevelProfilingTask(task)) {
            tidName = CLUSTERS_THREAD_NAME;
        } else if (isVariantLevelProfilingTask(task)) {
            tidName = createClusterThreadName(getClusterFromName(task.name));
        }

        ted.tid = findTid(tidName);
        ted.timestamp = task.start_time_ns;
        ted.duration = task.duration_ns;
        out_stream << ted;
    }

    ted.category = "Layer";
    for (auto& layer : layerProfiling) {
        ted.name = layer.name;
        ted.tid = findTid(EXEC_SUM_THREAD_NAME);
        ted.timestamp = layer.start_time_ns;
        ted.duration = layer.dpu_ns + layer.sw_ns + layer.dma_ns;
        out_stream << ted;

        ted.tid = findTid(LAYER_THREAD_NAME);
        ted.duration = layer.duration_ns;
        out_stream << ted;
    }

    // Last item so far has a comma in the end. If the stream was std::cout this comma can't be removed.
    // Adding metadata items in the end to make JSON correct.
    out_stream << std::string(R"({"name": "process_name", "ph": "M", "pid": )") << PID << R"(, "tid": )" << TID
               << R"(, "args": {"name" : "Inference"}},)" << std::endl;
    for (size_t tid = 0; tid < tidNames.size(); tid++) {
        out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )" << tid + 2
                   << R"(, "args": {"name" : ")" << tidNames[tid] << R"("}})";
        if (tid + 1 != tidNames.size()) {
            out_stream << ",";
        }
        out_stream << std::endl;
    }
    out_stream << "]";
    if (format == TimeUnitFormat::NS) {
        out_stream << "," << std::endl << "\"displayTimeUnit\": \"ns\"" << std::endl;
    }
    out_stream << "}" << std::endl;
}

void vpux::profiling::outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, uint64_t>& blob,
                                   const std::pair<const uint8_t*, uint64_t>& profiling, const std::string& filename,
                                   TimeUnitFormat format, VerbosityLevel verbosity) {
    if (filename.empty()) {
        streamWriter(profilingType, blob, profiling, std::cout, format, verbosity);
    } else {
        std::ofstream outfile;
        outfile.open(filename, std::ios::out | std::ios::trunc);
        if (outfile.is_open()) {
            streamWriter(profilingType, blob, profiling, outfile, format, verbosity);
            outfile.close();
        } else {
            std::cerr << "Can't write result into " << filename << std::endl;
        }
    }
}
