//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_profiling.h"
#include "zero_utils.h"

namespace vpux {
namespace zeroProfiling {

bool ProfilingPool::create() {
    auto ret = _graph_profiling_ddi_table_ext->pfnProfilingPoolCreate(_graph_handle, _profiling_count, &_handle);
    return ((ZE_RESULT_SUCCESS == ret) && (_handle != nullptr));
}

ProfilingPool::~ProfilingPool() {
    if (_handle) {
        _graph_profiling_ddi_table_ext->pfnProfilingPoolDestroy(_handle);
    }
}

void Profiling::create(const ze_graph_profiling_pool_handle_t& profiling_pool) {
    zeroUtils::throwOnFail("pfnProfilingQueryCreate",
                           _graph_profiling_ddi_table_ext->pfnProfilingQueryCreate(profiling_pool, _index, &_handle));
}

void Profiling::queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData) {
    if (_handle && pSize) {
        zeroUtils::throwOnFail("pfnProfilingQueryGetData", _graph_profiling_ddi_table_ext->pfnProfilingQueryGetData(
                                                                   _handle, profilingType, pSize, pData));
    }
}

void Profiling::getProfilingProperties(ze_device_profiling_data_properties_t* properties) {
    if (_handle && properties) {
        zeroUtils::throwOnFail(
                "getProfilingProperties",
                _graph_profiling_ddi_table_ext->pfnDeviceGetProfilingDataProperties(_device_handle, properties));
    }
}

Profiling::~Profiling() {
    if (_handle) {
        _graph_profiling_ddi_table_ext->pfnProfilingQueryDestroy(_handle);
    }
}

}  // namespace zeroProfiling
}  // namespace vpux
