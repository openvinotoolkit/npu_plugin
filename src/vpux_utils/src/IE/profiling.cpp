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

using namespace vpux;
namespace ie = InferenceEngine;

std::map<std::string, ie::InferenceEngineProfileInfo> vpux::convertProfilingLayersToIEInfo(
        std::vector<profiling_layer_info>& layerInfo) {
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