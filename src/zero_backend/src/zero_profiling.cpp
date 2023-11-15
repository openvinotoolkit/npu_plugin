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

namespace ie = InferenceEngine;
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

LayerStatistics ProfilingQuery::getLayerStatistics(ie::VPUXConfigParams::CompilerType compiler_type,
                                                   const std::vector<char>& blob) {
    verifyProfilingProperties();
    ProfilingFormat format = ProfilingFormat::NONE;
    std::ofstream outFile = openProfilingStream(&format);

    std::vector<LayerInfo> layerProfiling;

    if (compiler_type != ie::VPUXConfigParams::CompilerType::MLIR) {
        if (outFile.is_open()) {
            if (format != ProfilingFormat::RAW) {
                const auto taskProfiling = getData<TaskInfo>();
                layerProfiling = getData<LayerInfo>();
                saveProfilingDataToFile(format, outFile, layerProfiling, taskProfiling);
            } else {
                std::vector<uint8_t> rawBytes = getData<uint8_t>();
                saveRawDataToFile(rawBytes.data(), rawBytes.size(), outFile);
                layerProfiling = getData<LayerInfo>();
            }
        } else {
            layerProfiling = getData<LayerInfo>();
        }
    } else {
        // Process raw profiling data on the application side
        std::vector<uint8_t> rawBytes = getData<uint8_t>();
        const uint8_t* blob_data = reinterpret_cast<const uint8_t*>(blob.data());
        if (outFile.is_open()) {
            if (format != ProfilingFormat::RAW) {
                std::vector<TaskInfo> taskProfiling = getTaskInfo(blob_data, blob.size(), rawBytes.data(),
                                                                  rawBytes.size(), TaskType::ALL, VerbosityLevel::HIGH);
                layerProfiling = getLayerInfo(blob_data, blob.size(), rawBytes.data(), rawBytes.size());
                saveProfilingDataToFile(format, outFile, layerProfiling, taskProfiling);
            } else {
                saveRawDataToFile(rawBytes.data(), rawBytes.size(), outFile);
                layerProfiling = getLayerInfo(blob_data, blob.size(), rawBytes.data(), rawBytes.size());
            }
        } else {
            layerProfiling = getLayerInfo(blob_data, blob.size(), rawBytes.data(), rawBytes.size());
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

void ProfilingQuery::verifyProfilingProperties() {
    if (!_handle) {
        IE_THROW() << "Can't get profiling statistics because profiling is disabled.";
    }
    const auto stringifyVersion = [](auto version) -> std::string {
        return std::to_string(ZE_MAJOR_VERSION(version)) + "." + std::to_string(ZE_MINOR_VERSION(version));
    };

    ze_device_profiling_data_properties_t profProp;
    getProfilingProperties(&profProp);
    const auto currentProfilingVersion = ze_profiling_data_ext_version_t::ZE_PROFILING_DATA_EXT_VERSION_CURRENT;

    if (ZE_MAJOR_VERSION(profProp.extensionVersion) != ZE_MAJOR_VERSION(currentProfilingVersion)) {
        IE_THROW() << "Unsupported VPU driver."
                   << "Profiling API version: plugin: " << stringifyVersion(currentProfilingVersion)
                   << ", driver: " << stringifyVersion(profProp.extensionVersion);
    }
    // Now currentProfilingVersion minor version is 0, so next branch looks like "0 > (version & 0xFFFF)", what now is
    // always false. Overriding coverity warning
    /* coverity[result_independent_of_operands] */
    if (ZE_MINOR_VERSION(currentProfilingVersion) > ZE_MINOR_VERSION(profProp.extensionVersion)) {
        auto log = vpux::Logger::global().nest("ZeroProfilingQuery", 0);
        log.warning("Outdated VPU driver detected. Some features might not be available! "
                    "Profiling API version: plugin: {0}, driver: {1}",
                    stringifyVersion(currentProfilingVersion), stringifyVersion(profProp.extensionVersion));
    }
}

}  // namespace zeroProfiling
}  // namespace vpux
