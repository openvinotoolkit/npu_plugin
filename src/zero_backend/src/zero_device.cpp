//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/itt.hpp"
#include "ze_api.h"

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"
#include "zero_infer_request.h"

using namespace vpux;

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = std::make_shared<ZeroAllocator>(_driver_handle);
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Device::createExecutor");
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext,
                                          _graph_profiling_ddi_table_ext, networkDescription, config);
}

std::string ZeroDevice::getName() const {
    ze_device_properties_t properties;
    auto res = zeDeviceGetProperties(_device_handle, &properties);
    if (res != ZE_RESULT_SUCCESS) {
        IE_THROW() << "Error obtaining device name: err code " << res;
    }

//    KMD is setting usDeviceID from VpuFamilyID.h
#define IVPU_MYRIADX_DEVICE_ID 0x6200   // MyriadX device
#define IVPU_KEEMBAY_DEVICE_ID 0x6240   // KeemBay device
#define IVPU_VPUX37XX_DEVICE_ID 0x7D1D  // VPUX37XX device
#define IVPU_VPUX4000_DEVICE_ID 0x643E  // VPUX4000 device

    std::string name;
    switch (properties.deviceId) {
    case IVPU_MYRIADX_DEVICE_ID:
        name = "2700";
        break;
    case IVPU_KEEMBAY_DEVICE_ID:
        name = "3700";
        break;
    case IVPU_VPUX37XX_DEVICE_ID:
        name = "3720";
        break;
    case IVPU_VPUX4000_DEVICE_ID:
        // TODO to be changed
        name = "3720";
        break;
    default:
        name = "Unknown";
    }

    return name + ".0";
}

InferRequest::Ptr ZeroDevice::createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                                 const InferenceEngine::OutputsDataMap& networkOutputs,
                                                 const Executor::Ptr& executor, const Config& config,
                                                 const std::string& netName,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                 const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    return std::make_shared<ZeroInferRequest>(networkInputs, networkOutputs, executor, config, netName, parameters,
                                              results, allocator);
}
