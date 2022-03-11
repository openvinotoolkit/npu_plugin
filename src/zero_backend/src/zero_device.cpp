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
                                          networkDescription, config);
}

std::string ZeroDevice::getName() const {
    ze_device_properties_t properties;
    auto res = zeDeviceGetProperties(_device_handle, &properties);
    if (res != ZE_RESULT_SUCCESS) {
        IE_THROW() << "Error obtaining device name: err code " << res;
    }

//    KMD is setting usDeviceID from VpuFamilyID.h
#define IVPU_MYRIADX_DEVICE_ID 0x6200     // MyriadX device
#define IVPU_KEEMBAY_DEVICE_ID 0x6240     // KeemBay device
#define IVPU_METEORLAKE_DEVICE_ID 0x7D1D  // MeteorLake device
#define IVPU_LUNARLAKE_DEVICE_ID 0x643E   // LunarLake device

    std::string name;
    switch (properties.deviceId) {
    case IVPU_MYRIADX_DEVICE_ID:
        name = "2700";
        break;
    case IVPU_KEEMBAY_DEVICE_ID:
        name = "3700";
        break;
    case IVPU_METEORLAKE_DEVICE_ID:
        name = "3720";
        break;
    case IVPU_LUNARLAKE_DEVICE_ID:
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
