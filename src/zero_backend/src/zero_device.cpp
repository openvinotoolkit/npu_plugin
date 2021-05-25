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

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"

using namespace vpux;

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = std::make_shared<ZeroAllocator>(_driver_handle);
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    _config.parseFrom(config);
    std::shared_ptr<Executor> result;

    if (_config.ze_syncType() == InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE) {
        result = std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>>(
                _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext,
                networkDescription, _config);
    } else {
        result = std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>(
                _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext,
                networkDescription, _config);
    }

    return result;
}

std::string ZeroDevice::getName() const { return std::string("VPU-0"); }
