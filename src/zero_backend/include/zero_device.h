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

#include <ie_allocator.hpp>
#include <memory>
#include <vpux.hpp>
#include <vpux_compiler.hpp>

#include "ze_api.h"
#include "ze_fence_ext.h"
#include "ze_graph_ext.h"

#include "zero_config.h"
#include "zero_private_config.h"


namespace vpux {
class ZeroDevice : public IDevice {
    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_fence_dditable_ext_t* _fence_ddi_table_ext = nullptr;

    ZeroConfig _config;

public:
    ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
               ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext)
            : _driver_handle(driver),
              _device_handle(device),
              _context(context),
              _graph_ddi_table_ext(graph_ddi_table_ext),
              _fence_ddi_table_ext(fence_ddi_table_ext) { }

    std::shared_ptr<Allocator> getAllocator() const override;

    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    std::string getName() const override;

    ZeroDevice& operator=(const ZeroDevice&) = default;
    ZeroDevice(const ZeroDevice&) = default;
};
}  // namespace vpux
