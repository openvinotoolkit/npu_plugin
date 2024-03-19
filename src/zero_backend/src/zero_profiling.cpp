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

#include <fstream>

namespace vpux {
namespace zeroProfiling {

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

LayerStatistics ProfilingQuery::getLayerStatistics(InferenceEngine::VPUXConfigParams::CompilerType compiler_type,
                                                   const std::vector<char>& blob) {
    verifyProfilingProperties();
    ProfilingFormat format = ProfilingFormat::NONE;
    std::ofstream outFile = openProfilingStream(&format);

    std::vector<LayerInfo> layerProfiling;

    if (compiler_type != InferenceEngine::VPUXConfigParams::CompilerType::MLIR) {
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
                std::vector<TaskInfo> taskProfiling =
                        getTaskInfo(blob_data, blob.size(), rawBytes.data(), rawBytes.size(), VerbosityLevel::HIGH);
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
        OPENVINO_THROW("Can't get profiling statistics because profiling is disabled.");
    }
    const auto stringifyVersion = [](auto version) -> std::string {
        return std::to_string(ZE_MAJOR_VERSION(version)) + "." + std::to_string(ZE_MINOR_VERSION(version));
    };

    ze_device_profiling_data_properties_t profProp;
    getProfilingProperties(&profProp);
    const auto currentProfilingVersion = ze_profiling_data_ext_version_t::ZE_PROFILING_DATA_EXT_VERSION_CURRENT;

    if (ZE_MAJOR_VERSION(profProp.extensionVersion) != ZE_MAJOR_VERSION(currentProfilingVersion)) {
        OPENVINO_THROW("Unsupported VPU driver.",
                       "Profiling API version: plugin: ", stringifyVersion(currentProfilingVersion),
                       ", driver: ", stringifyVersion(profProp.extensionVersion));
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

VpuInferStatistics VpuInferProfiling::getVpuInferStatistics() const {
    VpuInferStatistics vpuPerfCounts;

    /// if the log isn't full/rolled over yet = skip reporting empty logs
    uint32_t stat_cnt = (_vpu_infer_stats_cnt < _vpu_infer_log_maxsize) ? _vpu_infer_stats_cnt : _vpu_infer_log_maxsize;
    if (stat_cnt != 0 && _loglevel >= LogLevel::Warning) {
        /// Populate vpuinferstatistics vector
        for (unsigned i = 0; i < stat_cnt; i++) {
            ov::ProfilingInfo info;

            info.status = ov::ProfilingInfo::Status::EXECUTED;
            info.real_time = std::chrono::microseconds(convertCCtoUS(_vpu_infer_duration_log[i]));
            info.cpu_time = std::chrono::microseconds(convertCCtoUS(_vpu_infer_duration_log[i]));
            info.node_name = std::to_string(i);
            info.exec_type = "INFER_REQ";
            info.node_type = "INFER_REQ";

            vpuPerfCounts.push_back(info);
        }
    }

    /// sanity check to avoid division by 0
    if (_vpu_infer_stats_cnt == 0) {
        return {};
    }

    /// Add final statistics
    ov::ProfilingInfo info_avg = {
            ov::ProfilingInfo::Status::EXECUTED,
            std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_accu_cc / _vpu_infer_stats_cnt)),
            std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_accu_cc / _vpu_infer_stats_cnt)),
            "AVG",
            "AVG",
            "AVG"};
    vpuPerfCounts.push_back(info_avg);
    ov::ProfilingInfo info_min = {ov::ProfilingInfo::Status::EXECUTED,
                                  std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_min_cc)),
                                  std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_min_cc)),
                                  "MIN",
                                  "MIN",
                                  "MIN"};
    vpuPerfCounts.push_back(info_min);
    ov::ProfilingInfo info_max = {ov::ProfilingInfo::Status::EXECUTED,
                                  std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_max_cc)),
                                  std::chrono::microseconds(convertCCtoUS(_vpu_infer_stats_max_cc)),
                                  "MAX",
                                  "MAX",
                                  "MAX"};
    vpuPerfCounts.push_back(info_max);
    return vpuPerfCounts;
}

VpuInferProfiling::VpuInferProfiling(ze_context_handle_t context, ze_device_handle_t device_handle, LogLevel loglevel)
        : _context(context), _device_handle(device_handle), _loglevel(loglevel), _logger("InferProfiling", loglevel) {
    /// Fetch and store the device timer resolution
    _dev_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_device_handle, &_dev_properties));
    /// Request mem allocations
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                     ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED};
    zeroUtils::throwOnFail("zeMemAllocHost",
                           zeMemAllocHost(_context, &desc, sizeof(uint64_t), 64,
                                          &vpu_ts_infer_start));  // align to 64 bytes to match npu l2 cache line size
    zeroUtils::throwOnFail("zeMemAllocHost",
                           zeMemAllocHost(_context, &desc, sizeof(uint64_t), 64,
                                          &vpu_ts_infer_end));  // alight to 64 bytes to match npu l2 cache line size
}

void VpuInferProfiling::sampleVpuTimestamps() {
    int64_t infer_duration_cc = (int64_t)((*(uint64_t*)vpu_ts_infer_end - *(uint64_t*)vpu_ts_infer_start));
    /// Update extremas
    if (infer_duration_cc < _vpu_infer_stats_min_cc)
        _vpu_infer_stats_min_cc = infer_duration_cc;
    if (infer_duration_cc > _vpu_infer_stats_max_cc)
        _vpu_infer_stats_max_cc = infer_duration_cc;
    _vpu_infer_stats_accu_cc += infer_duration_cc;
    _vpu_infer_stats_cnt++;
    /// only log individual infer durations if requested
    if (_loglevel >= LogLevel::Warning) {
        _vpu_infer_duration_log[_vpu_infer_logidx++] = infer_duration_cc;
        if (_vpu_infer_logidx >= _vpu_infer_log_maxsize)
            _vpu_infer_logidx = 0;
    }
}

int64_t VpuInferProfiling::convertCCtoUS(int64_t val_cc) const {
    return (int64_t)(val_cc * 1000 * 1000 / _dev_properties.timerResolution);
}

VpuInferProfiling::~VpuInferProfiling() {
    /// deallocate vpu_ts_infer_start and vpu_ts_infer_end, allocated externally by ze driver
    if (vpu_ts_infer_start != nullptr) {
        auto ze_ret = zeMemFree(_context, vpu_ts_infer_start);
        if (ZE_RESULT_SUCCESS != ze_ret) {
            _logger.error("zeMemFree on vpu_ts_infer_start failed {0:X+}", uint64_t(ze_ret));
        }
    }
    if (vpu_ts_infer_end != nullptr) {
        auto ze_ret = zeMemFree(_context, vpu_ts_infer_end);
        if (ZE_RESULT_SUCCESS != ze_ret) {
            _logger.error("zeMemFree on vpu_ts_infer_end failed {0:X+}", uint64_t(ze_ret));
        }
    }
}

}  // namespace zeroProfiling
}  // namespace vpux
