//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ze_api.h"
#include "ze_graph_ext.h"

#include <vpux.hpp>
#include <vpux_compiler.hpp>
#include "vpux/utils/core/logger.hpp"

#include <ie_allocator.hpp>

#include <memory>

namespace vpux {
class ZeroDevice : public IDevice {
public:
    ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
               ze_graph_dditable_ext_t* graph_ddi_table_ext,
               ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext);

    std::shared_ptr<Allocator> getAllocator() const override;

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                             const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    uint64_t getTotalMemSize() const override;
    uint32_t getDriverVersion() const override;

    IInferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                          const InferenceEngine::OutputsDataMap& networkOutputs,
                                          const Executor::Ptr& executor, const Config& config,
                                          const std::string& netName,
                                          const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                          const std::vector<std::shared_ptr<const ov::Node>>& results,
                                          const vpux::NetworkIOVector& networkStatesInfo,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator) override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

private:
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    uint64_t totalMemSize = 0;
    std::string fullDeviceName;

    uint32_t _group_ordinal;

    Logger log;
};
}  // namespace vpux
