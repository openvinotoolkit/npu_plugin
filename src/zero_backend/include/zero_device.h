//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ze_api.h"
#include "ze_graph_ext.h"

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux_compiler.hpp"
#include "zero_types.h"
#include "zero_utils.h"

namespace vpux {
class ZeroDevice : public IDevice {
public:
    ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
               ze_graph_dditable_ext_curr_t* graph_ddi_table_ext,
               ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext);

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr networkDescription,
                                             const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    uint64_t getAllocMemSize() const override;
    uint64_t getTotalMemSize() const override;
    uint32_t getDriverVersion() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> compiledModel,
            const std::shared_ptr<const NetworkDescription> networkDescription, const Executor::Ptr executor,
            const Config& config) override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

private:
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    std::string fullDeviceName;

    uint32_t _group_ordinal;

    Logger log;
};
}  // namespace vpux
