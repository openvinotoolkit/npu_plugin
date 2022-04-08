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

using namespace vpux;
using namespace vpux::profiling;
namespace ie = InferenceEngine;

enum ThreadId { TID_DMA = 2, TID_DPU, TID_SW, TID_LAYER_SUM_OF_DURATIONS, TID_LAYER_DURATION };

static constexpr int PID = 1, TID = 1;

static const std::map<TaskInfo::ExecType, std::string> enumToStr = {
        {TaskInfo::ExecType::NONE, "NONE"},
        {TaskInfo::ExecType::DPU, "DPU"},
        {TaskInfo::ExecType::SW, "SW"},
        {TaskInfo::ExecType::DMA, "DMA"},
};

static const std::map<TaskInfo::ExecType, ThreadId> enumToThreadId = {
        {TaskInfo::ExecType::DPU, ThreadId::TID_DPU},
        {TaskInfo::ExecType::SW, ThreadId::TID_SW},
        {TaskInfo::ExecType::DMA, ThreadId::TID_DMA},
};

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

static void printProfilingAsText(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                 std::ostream& out_stream) {
    std::vector<TaskInfo> taskProfiling = getTaskInfo(blobData, blobSize, profData, profSize, TaskType::ALL);

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

    std::vector<LayerInfo> layerProfiling = getLayerInfo(taskProfiling);
    uint64_t total_time = 0;
    for (auto& layer : layerProfiling) {
        out_stream << "Layer: " << std::setw(80) << layer.name << " DPU: " << std::setw(5) << layer.dpu_ns / 1000
                   << " SW: " << std::setw(5) << layer.sw_ns / 1000 << " DMA: " << std::setw(5) << layer.dma_ns / 1000
                   << "\tStart: " << layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    out_stream << "TotalTime: " << total_time / 1000 << "us, Real: " << last_time_ns / 1000 << "us" << std::endl;
}

static void printProfilingAsTraceEvent(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                       size_t profSize, std::ostream& out_stream) {
    struct TracingEventDesc ted;
    ted.pid = PID;
    std::vector<TaskInfo> taskProfiling = getTaskInfo(blobData, blobSize, profData, profSize, TaskType::ALL);

    out_stream << "{\"traceEvents\":[" << std::endl;

    for (auto& task : taskProfiling) {
        ted.name = task.name;
        ted.category = enumToStr.at(task.exec_type);
        ted.tid = enumToThreadId.at(task.exec_type);
        ted.timestamp = task.start_time_ns;
        ted.duration = task.duration_ns;
        out_stream << ted;
    }

    std::vector<LayerInfo> layerProfiling = getLayerInfo(blobData, blobSize, profData, profSize);
    ted.category = "Layer";
    for (auto& layer : layerProfiling) {
        ted.name = layer.name;
        ted.tid = ThreadId::TID_LAYER_SUM_OF_DURATIONS;
        ted.timestamp = layer.start_time_ns;
        ted.duration = layer.dpu_ns + layer.sw_ns + layer.dma_ns;
        out_stream << ted;

        ted.tid = ThreadId::TID_LAYER_DURATION;
        ted.duration = layer.duration_ns;
        out_stream << ted;
    }

    // Last item so far has a comma in the end. If the stream was std::cout this comma can't be removed.
    // Adding metadata items in the end to make JSON correct.
    out_stream << std::string(R"({"name": "process_name", "ph": "M", "pid": )") << PID << R"(, "tid": )" << TID
               << R"(, "args": {"name" : "Inference"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )"
               << ThreadId::TID_DMA << R"(, "args": {"name" : "DMA"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )"
               << ThreadId::TID_DPU << R"(, "args": {"name" : "DPU"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )"
               << ThreadId::TID_SW << R"(, "args": {"name" : "SW"}},)" << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )"
               << ThreadId::TID_LAYER_SUM_OF_DURATIONS << R"(, "args": {"name" : "Sum of execution times"}},)"
               << std::endl;
    out_stream << std::string(R"({"name": "thread_name", "ph": "M", "pid": )") << PID << R"(, "tid": )"
               << ThreadId::TID_LAYER_DURATION << R"(, "args": {"name" : "Layer execution time"}})" << std::endl;

    out_stream << "]," << std::endl << "\"displayTimeUnit\": \"ns\"" << std::endl << "}" << std::endl;
}

static void streamWriter(const OutputType profilingType, const std::pair<const uint8_t*, uint64_t>& blob,
                         const std::pair<const uint8_t*, uint64_t>& profiling, std::ostream& output) {
    const auto blobData = blob.first;
    const auto blobSize = blob.second;
    const auto profilingData = profiling.first;
    const auto profilingSize = profiling.second;
    switch (profilingType) {
    case OutputType::TEXT:
        printProfilingAsText(blobData, blobSize, profilingData, profilingSize, output);
        break;
    case OutputType::JSON:
        printProfilingAsTraceEvent(blobData, blobSize, profilingData, profilingSize, output);
        break;
    case OutputType::NONE:
        break;
    default:
        std::cerr << "Unsupported profiling output type." << std::endl;
        break;
    }
};

void vpux::profiling::outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, uint64_t>& blob,
                                   const std::pair<const uint8_t*, uint64_t>& profiling, const std::string& filename) {
    if (filename.empty()) {
        streamWriter(profilingType, blob, profiling, std::cout);
    } else {
        std::ofstream outfile;
        outfile.open(filename, std::ios::out | std::ios::trunc);
        if (outfile.is_open()) {
            streamWriter(profilingType, blob, profiling, outfile);
            outfile.close();
        } else {
            std::cerr << "Can't write result into " << filename << std::endl;
        }
    }
}
