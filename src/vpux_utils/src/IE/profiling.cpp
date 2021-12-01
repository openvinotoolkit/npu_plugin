//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/utils/IE/profiling.hpp"

#include <iomanip>
#include <iostream>

using namespace vpux;
namespace ie = InferenceEngine;

std::map<std::string, ie::InferenceEngineProfileInfo> vpux::convertProfilingLayersToIEInfo(
        std::vector<ProfilingLayerInfo>& layerInfo) {
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
        auto typeLen = sizeof(info.exec_type) / sizeof(info.exec_type[0]);
        if (layer.sw_ns > 0) {
            info.cpu_uSec = layer.sw_ns / 1000;
            strncpy(info.exec_type, "SW", typeLen - 1);
        } else if (layer.dpu_ns > 0) {
            info.cpu_uSec = layer.dpu_ns / 1000;
            strncpy(info.exec_type, "DPU", typeLen - 1);
        } else {
            info.cpu_uSec = 0;
            strncpy(info.exec_type, "DMA", typeLen - 1);
        }
        typeLen = sizeof(info.layer_type) / sizeof(info.layer_type[0]);
        strncpy(info.layer_type, layer.layer_type, typeLen - 1);
        perfCounts[name] = info;
    }

    return perfCounts;
}

void vpux::printProfiling(const void* data, size_t data_len, const void* output, size_t output_len) {
    std::vector<vpux::ProfilingTaskInfo> taskProfiling;
    vpux::getTaskProfilingInfo(data, data_len, output, output_len, taskProfiling, vpux::ProfilingTaskType::ALL);

    uint64_t last_time_ns = 0;
    for (auto& task : taskProfiling) {
        std::string exec_type_str;
        switch (task.exec_type) {
        case vpux::ProfilingTaskInfo::exec_type_t::DMA:
            exec_type_str = "DMA";
            std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name
                      << "\tTime: " << (float)task.duration_ns / 1000 << "\tStart: " << task.start_time_ns / 1000
                      << std::endl;
            break;
        case vpux::ProfilingTaskInfo::exec_type_t::DPU:
            exec_type_str = "DPU";
            std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name
                      << "\tTime: " << (float)task.duration_ns / 1000 << "\tStart: " << task.start_time_ns / 1000
                      << std::endl;
            break;
        case vpux::ProfilingTaskInfo::exec_type_t::SW:
            exec_type_str = "SW";
            std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name
                      << "\tTime: " << (float)task.duration_ns / 1000 << "\tCycles:" << task.active_cycles << "("
                      << task.stall_cycles << ")" << std::endl;
            break;
        default:
            break;
        }

        uint64_t task_end_time_ns = task.start_time_ns + task.duration_ns;
        if (last_time_ns < task_end_time_ns) {
            last_time_ns = task_end_time_ns;
        }
    }

    std::vector<vpux::ProfilingLayerInfo> layerProfiling;
    vpux::getLayerProfilingInfo(data, data_len, output, output_len, layerProfiling);
    uint64_t total_time = 0;
    for (auto& layer : layerProfiling) {
        std::cout << "Layer: " << std::setw(40) << layer.name << " DPU: " << std::setw(5) << layer.dpu_ns / 1000
                  << " SW: " << std::setw(5) << layer.sw_ns / 1000 << " DMA: " << std::setw(5) << layer.dma_ns / 1000
                  << "\tStart: " << layer.start_time_ns / 1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    std::cout << "TotalTime: " << total_time / 1000 << "us, Real: " << last_time_ns / 1000 << "us" << std::endl;
}