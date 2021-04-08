//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "zero_backend.h"

#include <description_buffer.hpp>
#include <iostream>
#include <vector>

#include "zero_device.h"

using namespace vpux;

namespace {
class ZeroDevicesSingleton {
    ZeroDevicesSingleton() {
        if (ZE_RESULT_SUCCESS != zeInit(0)) {
            std::cerr << "ZeroDevicesSingleton zeInit failed\n";
            return;
        }

        ze_driver_handle_t driver_handle = nullptr;
        ze_device_handle_t device_handle = nullptr;
        ze_context_handle_t context = nullptr;

        
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
        ze_fence_dditable_ext_t* _fence_ddi_table_ext = nullptr;

        uint32_t drivers = 0;
        if (ZE_RESULT_SUCCESS != zeDriverGet(&drivers, nullptr)) {
            std::cerr << "ZeroDevicesSingleton zeDriverGet count failed\n";
            return;
        }
        std::vector<ze_driver_handle_t> all_drivers(drivers);
        if (ZE_RESULT_SUCCESS != zeDriverGet(&drivers, all_drivers.data())) {
            std::cerr << "ZeroDevicesSingleton zeDriverGet get failed\n";
            return;
        }
        // Get our target driver
        for (uint32_t i = 0; i < drivers; ++i) {
            // arbitrary test at this point
            if (i == drivers - 1) {
                driver_handle = all_drivers[i];
            }
        }
        
        // Load our graph extension
        if (ZE_RESULT_SUCCESS != zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_graph",
                                                                     reinterpret_cast<void**>(&_graph_ddi_table_ext))) {
            std::cerr << "ZeroDevicesSingleton zeDriverGetExtensionFunctionAddress failed\n";
            return;
        }

        // Load our fence extension
        if (ZE_RESULT_SUCCESS != zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_fence",
                                                                     reinterpret_cast<void**>(&_fence_ddi_table_ext))) {
            std::cerr << "ZeroDevicesSingleton zeDriverGetExtensionFunctionAddress failed\n";
            return;
        }

        uint32_t device_count = 1;
        // Get our target device
        if (ZE_RESULT_SUCCESS != zeDeviceGet(driver_handle, &device_count, &device_handle)) {
            std::cerr << "ZeroDevicesSingleton zeDeviceGet failed\n";
            return;
        }

        ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0};

         if (ZE_RESULT_SUCCESS != zeContextCreate(driver_handle, &context_desc, &context)) {
            std::cerr << "ZeroDevicesSingleton zeContextCreate failed\n";
            return;
        }

        auto device = std::make_shared<ZeroDevice>(driver_handle, device_handle, context,
                                                   _graph_ddi_table_ext, _fence_ddi_table_ext);
        devices.emplace(std::make_pair(device->getName(), device));
    }

    ~ZeroDevicesSingleton() = default;
    ZeroDevicesSingleton(const ZeroDevicesSingleton&) = delete;
    ZeroDevicesSingleton& operator=(const ZeroDevicesSingleton&) = delete;

    std::map<std::string, std::shared_ptr<IDevice>> devices;

public:
    static const std::map<std::string, std::shared_ptr<IDevice>>& getInstanceDevices() {
        static ZeroDevicesSingleton instance;
        return instance.devices;
    }
};
}  // namespace

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice() const {
    if (ZeroDevicesSingleton::getInstanceDevices().size())
        return ZeroDevicesSingleton::getInstanceDevices().begin()->second;
    else
        return {};
}
const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    const auto& devices = ZeroDevicesSingleton::getInstanceDevices();
    std::vector<std::string> devicesNames;
    std::for_each(devices.cbegin(), devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });

    return devicesNames;
}

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new vpux::ZeroEngineBackend();
        return InferenceEngine::StatusCode::OK;
    } catch (std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
