//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <openvino/runtime/profiling_info.hpp>
#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/core/logger.hpp"

#include <climits>
#include <map>

namespace vpux {
namespace zeroProfiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

constexpr uint32_t POOL_SIZE = 1;

struct ProfilingPool {
    ProfilingPool(ze_graph_handle_t graph_handle, uint32_t profiling_count,
                  ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext)
            : _graph_handle(graph_handle),
              _profiling_count(profiling_count),
              _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext) {
    }
    ProfilingPool(const ProfilingPool&) = delete;
    ProfilingPool& operator=(const ProfilingPool&) = delete;
    bool create();

    ~ProfilingPool();

    ze_graph_handle_t _graph_handle;
    const uint32_t _profiling_count;
    ze_graph_profiling_pool_handle_t _handle = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;
};

struct ProfilingQuery {
    ProfilingQuery(uint32_t index, ze_device_handle_t device_handle,
                   ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext)
            : _index(index),
              _device_handle(device_handle),
              _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext) {
    }
    ProfilingQuery(const ProfilingQuery&) = delete;
    ProfilingQuery& operator=(const ProfilingQuery&) = delete;
    void create(const ze_graph_profiling_pool_handle_t& profiling_pool);
    ze_graph_profiling_query_handle_t getHandle() const {
        return _handle;
    }
    LayerStatistics getLayerStatistics(InferenceEngine::VPUXConfigParams::CompilerType compiler_type,
                                       const std::vector<char>& blob);
    ~ProfilingQuery();

private:
    void queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData);
    template <class ProfilingData>
    std::vector<ProfilingData> getData();
    void getProfilingProperties(ze_device_profiling_data_properties_t* properties);
    void verifyProfilingProperties();

    const uint32_t _index;
    ze_device_handle_t _device_handle;
    ze_graph_profiling_query_handle_t _handle = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;
};

using VpuInferStatistics = std::vector<ov::ProfilingInfo>;

struct VpuInferProfiling final {
    explicit VpuInferProfiling(ze_context_handle_t context, ze_device_handle_t device_handle, LogLevel loglevel);
    VpuInferProfiling(const VpuInferProfiling&) = delete;
    VpuInferProfiling& operator=(const VpuInferProfiling&) = delete;
    VpuInferProfiling(VpuInferProfiling&&) = delete;
    VpuInferProfiling& operator=(VpuInferProfiling&&) = delete;

    void sampleVpuTimestamps();
    VpuInferStatistics getVpuInferStatistics() const;

    ~VpuInferProfiling();

    /// Buffers allocated by ZE driver
    void* vpu_ts_infer_start = 0;
    void* vpu_ts_infer_end = 0;

private:
    ze_context_handle_t _context = nullptr;
    ze_device_handle_t _device_handle;
    LogLevel _loglevel;
    Logger _logger;
    ze_device_properties_t _dev_properties;
    int64_t _vpu_infer_stats_min_cc = LLONG_MAX;
    int64_t _vpu_infer_stats_max_cc = 0;
    int64_t _vpu_infer_stats_accu_cc = 0;
    uint32_t _vpu_infer_stats_cnt = 0;
    uint32_t _vpu_infer_logidx = 0;
    static const uint32_t _vpu_infer_log_maxsize = 1024;
    /// rolling buffer to store duration of last <_vpu_infer_log_maxsize number> infers
    int64_t _vpu_infer_duration_log[_vpu_infer_log_maxsize];

    /// Helper function to convert vpu clockcycles to usec
    int64_t convertCCtoUS(int64_t val_cc) const;
};

}  // namespace zeroProfiling
}  // namespace vpux
