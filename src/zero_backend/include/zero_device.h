//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include "ze_api.h"
#include "ze_graph_ext.h"

#include <vpux.hpp>
#include <vpux_compiler.hpp>

#include <ie_allocator.hpp>

#include <memory>

namespace vpux {
class ZeroDevice : public IDevice {
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;

public:
    ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
               ze_graph_dditable_ext_t* graph_ddi_table_ext)
            : _driver_handle(driver),
              _device_handle(device),
              _context(context),
              _graph_ddi_table_ext(graph_ddi_table_ext) {
    }

    std::shared_ptr<Allocator> getAllocator() const override;

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                             const Config& config) override;

    std::string getName() const override;

    InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                         const InferenceEngine::OutputsDataMap& networkOutputs,
                                         const Executor::Ptr& executor, const Config& config,
                                         const std::string& netName,
                                         const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                         const std::vector<std::shared_ptr<const ov::Node>>& results,
                                         const std::shared_ptr<InferenceEngine::IAllocator>& allocator) override;

    ZeroDevice& operator=(const ZeroDevice&) = default;
    ZeroDevice(const ZeroDevice&) = default;
};
}  // namespace vpux
