//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

namespace vpux {
namespace zeroProfiling {

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

struct Profiling {
    Profiling(uint32_t index, ze_device_handle_t device_handle,
              ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext)
            : _index(index),
              _device_handle(device_handle),
              _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext) {
    }
    Profiling(const Profiling&) = delete;
    Profiling& operator=(const Profiling&) = delete;
    void create(const ze_graph_profiling_pool_handle_t& profiling_pool);
    void queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData);
    void getProfilingProperties(ze_device_profiling_data_properties_t* properties);

    ~Profiling();

    const uint32_t _index;
    ze_device_handle_t _device_handle;
    ze_graph_profiling_query_handle_t _handle = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;
};

}  // namespace zeroProfiling
}  // namespace vpux
