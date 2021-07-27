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

#include "zero_backend.h"

#include <description_buffer.hpp>
#include <iostream>
#include <vector>

#include "zero_device.h"

using namespace vpux;

namespace {
class ZeroDevicesSingleton {
    ZeroDevicesSingleton() {
        auto result = zeInit(0);
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeInit failed 0x" << std::hex << uint64_t(result) << std::dec
                      << std::endl;
            return;
        }

        ze_driver_handle_t driver_handle = nullptr;
        ze_device_handle_t device_handle = nullptr;
        ze_context_handle_t context = nullptr;

        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;

        uint32_t drivers = 0;
        result = zeDriverGet(&drivers, nullptr);
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeDriverGet count failed 0x" << std::hex << uint64_t(result) << std::dec
                      << std::endl;
            return;
        }
        std::vector<ze_driver_handle_t> all_drivers(drivers);
        result = zeDriverGet(&drivers, all_drivers.data());
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeDriverGet get failed 0x" << std::hex << uint64_t(result) << std::dec
                      << std::endl;
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
        result = zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_graph",
                                                     reinterpret_cast<void**>(&_graph_ddi_table_ext));
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeDriverGetExtensionFunctionAddress failed 0x" << std::hex
                      << uint64_t(result) << std::dec << std::endl;
            return;
        }

        uint32_t device_count = 1;
        // Get our target device
        result = zeDeviceGet(driver_handle, &device_count, &device_handle);
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeDeviceGet failed 0x" << std::hex << uint64_t(result) << std::dec
                      << std::endl;
            return;
        }

        ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
        result = zeContextCreate(driver_handle, &context_desc, &context);
        if (ZE_RESULT_SUCCESS != result) {
            std::cerr << "ZeroDevicesSingleton zeContextCreate failed 0x" << std::hex << uint64_t(result) << std::dec
                      << std::endl;
            return;
        }

        auto device = std::make_shared<ZeroDevice>(driver_handle, device_handle, context, _graph_ddi_table_ext);
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

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice(const std::string& /*name*/) const {
    // TODO Add the search of the device by platform & slice
    return getDevice();
}

const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    const auto& devices = ZeroDevicesSingleton::getInstanceDevices();
    std::vector<std::string> devicesNames;
    std::for_each(devices.cbegin(), devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });

    return devicesNames;
}

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::ZeroEngineBackend>();
}
