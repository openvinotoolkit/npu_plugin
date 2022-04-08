//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_profiling.h"
#include "zero_utils.h"

#include "ze_api.h"
#include "ze_graph_profiling_ext.h"
#include "zero_profiling.h"

#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

#include <ie_common.h>

#include <fstream>
#include <string>

namespace vpux {
namespace zeroProfiling {

namespace IE = InferenceEngine;
using namespace vpux::profiling;

static_assert(sizeof(LayerInfo) == sizeof(ze_profiling_layer_info), "Profiling type mismtach");
static_assert(sizeof(TaskInfo) == sizeof(ze_profiling_task_info), "Profiling type mismtach");

/// @brief Type trait mapping from ZE data type to enum value
template <typename T>
struct ZeProfilingTypeId {};

template <>
struct ZeProfilingTypeId<ze_profiling_layer_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_LAYER_LEVEL;
};

template <>
struct ZeProfilingTypeId<LayerInfo> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_LAYER_LEVEL;
};

template <>
struct ZeProfilingTypeId<ze_profiling_task_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_TASK_LEVEL;
};

template <>
struct ZeProfilingTypeId<TaskInfo> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_TASK_LEVEL;
};

template <>
struct ZeProfilingTypeId<uint8_t> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_RAW;
};

std::string getEnvVar(const std::string& varName, bool capitalize, const std::string& defaultValue = "") {
    const char* rawValue = std::getenv(varName.c_str());
    std::string value = rawValue == nullptr ? defaultValue : rawValue;

    if (capitalize) {
        std::transform(value.begin(), value.end(), value.begin(), ::toupper);
    }
    return value;
}

enum class ProfilingFormat { NONE, JSON, TEXT, RAW };

ProfilingFormat getProfilingFormat(const std::string& format) {
    if (format == "JSON")
        return ProfilingFormat::JSON;
    if (format == "TEXT")
        return ProfilingFormat::TEXT;
    if (format == "RAW")
        return ProfilingFormat::RAW;
    return ProfilingFormat::NONE;
}

void saveProfilingDataToFile(ProfilingFormat format, std::ostream& outfile,
                             const std::vector<LayerInfo>& layerProfiling, const std::vector<TaskInfo>& taskProfiling) {
    const auto DEFAULT_VERBOSE_VALUE = "LOW";
    static const std::map<std::string, size_t> VERBOSITY_TO_NUM_FILTERS = {
            {DEFAULT_VERBOSE_VALUE, 2},
            {"MEDIUM", 1},
            {"HIGH", 0},
    };
    auto verbosityValue = getEnvVar("VPUX_PROFILING_VERBOSITY", true, DEFAULT_VERBOSE_VALUE);
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

void saveRawDataToFile(uint8_t* rawBuffer, size_t size, std::ostream& outfile) {
    outfile.write(reinterpret_cast<char*>(rawBuffer), size);
}

LayerStatistics convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo) {
    LayerStatistics perfCounts;
    int execution_index = 0;
    for (const auto& layer : layerInfo) {
        IE::InferenceEngineProfileInfo info;
        info.status = IE::InferenceEngineProfileInfo::EXECUTED;
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

bool ProfilingPool::create() {
    auto ret = _graph_profiling_ddi_table_ext->pfnProfilingPoolCreate(_graph_handle, _profiling_count, &_handle);
    return ((ZE_RESULT_SUCCESS == ret) && (_handle != nullptr));
}

ProfilingPool::~ProfilingPool() {
    if (_handle) {
        _graph_profiling_ddi_table_ext->pfnProfilingPoolDestroy(_handle);
    }
}

void ProfilingQuery::create(const ze_graph_profiling_pool_handle_t& profiling_pool) {
    zeroUtils::throwOnFail("pfnProfilingQueryCreate",
                           _graph_profiling_ddi_table_ext->pfnProfilingQueryCreate(profiling_pool, _index, &_handle));
}

LayerStatistics ProfilingQuery::getLayerStatistics(IE::VPUXConfigParams::CompilerType compiler_type,
                                                   const std::vector<char>& blob) {
    if (!(_handle)) {
        IE_THROW() << "Can't get profiling statistics because profiling is disabled.";
    }
    const auto supportedProfilingVersion = ZE_MAKE_VERSION(1, 0);

    ze_device_profiling_data_properties_t profProp;
    getProfilingProperties(&profProp);
    if (profProp.extensionVersion != supportedProfilingVersion) {
        IE_THROW() << "Profiling version mismatch. Probably you need to update the application.";
    }
    std::ofstream outFile;

    const auto printProfiling = getEnvVar("VPUX_PRINT_PROFILING", true);
    ProfilingFormat format = getProfilingFormat(printProfiling);

    if (format != ProfilingFormat::NONE) {
        const auto outFileName = getEnvVar("VPUX_PROFILING_OUTPUT_FILE", false);
        auto flags = std::ios::out | std::ios::trunc;
        if (format == ProfilingFormat::RAW) {
            flags |= std::ios::binary;
        }
        outFile.open(outFileName, flags);
        if (!outFile) {
            std::cerr << "Can't write result into " << outFileName << std::endl;
        }
    }

    std::vector<LayerInfo> layerProfiling;

    if (compiler_type != IE::VPUXConfigParams::CompilerType::MLIR) {
        layerProfiling = getData<LayerInfo>();
        if (outFile.is_open()) {
            if (format != ProfilingFormat::RAW) {
                const auto taskProfiling = getData<TaskInfo>();
                saveProfilingDataToFile(format, outFile, layerProfiling, taskProfiling);
            } else {
                std::vector<uint8_t> rawBytes = getData<uint8_t>();
                saveRawDataToFile(rawBytes.data(), rawBytes.size(), outFile);
            }
        }
    } else {
        // Process raw profiling data on the application side
        std::vector<uint8_t> rawBytes = getData<uint8_t>();
        const uint8_t* blob_data = reinterpret_cast<const uint8_t*>(blob.data());
        layerProfiling = getLayerInfo(blob_data, blob.size(), rawBytes.data(), rawBytes.size());
        if (outFile.is_open()) {
            if (format != ProfilingFormat::RAW) {
                std::vector<TaskInfo> taskProfiling = getTaskInfo(blob_data, blob.size(), rawBytes.data(),
                                                                  rawBytes.size(), TaskType::ALL, VerbosityLevel::HIGH);
                saveProfilingDataToFile(format, outFile, layerProfiling, taskProfiling);
            } else {
                saveRawDataToFile(rawBytes.data(), rawBytes.size(), outFile);
            }
        }
    }
    return convertLayersToIeProfilingInfo(layerProfiling);
}

ProfilingQuery::~ProfilingQuery() {
    if (_handle) {
        _graph_profiling_ddi_table_ext->pfnProfilingQueryDestroy(_handle);
    }
}

void ProfilingQuery::queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData) {
    if (_handle && pSize) {
        zeroUtils::throwOnFail("pfnProfilingQueryGetData", _graph_profiling_ddi_table_ext->pfnProfilingQueryGetData(
                                                                   _handle, profilingType, pSize, pData));
    }
}

template <class ProfilingData>
std::vector<ProfilingData> ProfilingQuery::getData() {
    ze_graph_profiling_type_t type = ZeProfilingTypeId<ProfilingData>::value;
    uint32_t size = 0;

    // Obtain the size of the buffer
    queryGetData(type, &size, nullptr);

    IE_ASSERT(size % sizeof(ProfilingData) == 0);

    // Allocate enough memory and copy the buffer
    std::vector<ProfilingData> profilingData(size / sizeof(ProfilingData));
    queryGetData(type, &size, reinterpret_cast<uint8_t*>(profilingData.data()));
    return profilingData;
}

void ProfilingQuery::getProfilingProperties(ze_device_profiling_data_properties_t* properties) {
    if (_handle && properties) {
        zeroUtils::throwOnFail(
                "getProfilingProperties",
                _graph_profiling_ddi_table_ext->pfnDeviceGetProfilingDataProperties(_device_handle, properties));
    }
}

}  // namespace zeroProfiling
}  // namespace vpux
