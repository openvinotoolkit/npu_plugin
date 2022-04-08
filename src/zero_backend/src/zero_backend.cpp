//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_backend.h"

#include <description_buffer.hpp>
#include <vector>

#include "zero_device.h"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class ZeroStructsInitializer {
public:
    ZeroStructsInitializer();
    ~ZeroStructsInitializer();
    const std::map<std::string, std::shared_ptr<IDevice>>& getInstanceDevices() {
        return devices;
    }

private:
    ZeroStructsInitializer(const ZeroStructsInitializer&) = delete;
    ZeroStructsInitializer& operator=(const ZeroStructsInitializer&) = delete;

    static const ze_driver_uuid_t uuid;
    Logger log;

    std::map<std::string, std::shared_ptr<IDevice>> devices{};

    ze_context_handle_t context = nullptr;
};

const ze_driver_uuid_t ZeroStructsInitializer::uuid = {0x01, 0x7d, 0xe9, 0x31, 0x6b, 0x4d, 0x4f, 0xd4,
                                                       0xaa, 0x9b, 0x5b, 0xed, 0x77, 0xfc, 0x8e, 0x89};

ZeroStructsInitializer::ZeroStructsInitializer(): log(Logger::global().nest("ZeroStructsInitializer", 0)) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroStructsInitializer::ZeroStructsInitializer");
    auto result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeInit failed {0:X+}", uint64_t(result));
        return;
    }

    ze_driver_handle_t driver_handle = nullptr;
    ze_device_handle_t device_handle = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    uint32_t drivers = 0;
    result = zeDriverGet(&drivers, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeDriverGet count failed {0:X+}", uint64_t(result));
        return;
    }
    std::vector<ze_driver_handle_t> all_drivers(drivers);
    result = zeDriverGet(&drivers, all_drivers.data());
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeDriverGet get failed {0:X+}", uint64_t(result));
        return;
    }

    // Get our target driver
    ze_driver_properties_t props{};
    for (uint32_t i = 0; i < drivers; ++i) {
        zeDriverGetProperties(all_drivers[i], &props);

        if (memcmp(&props.uuid, &uuid, sizeof(uuid)) == 0) {
            driver_handle = all_drivers[i];
            break;
        }
    }
    if (driver_handle == nullptr) {
        log.warning("zeDriverGet failed to return VPU driver");
        return;
    }

    // Load our graph extension
    result = zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_graph",
                                                 reinterpret_cast<void**>(&_graph_ddi_table_ext));
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeDriverGetExtensionFunctionAddress failed {0:X+}", uint64_t(result));
        return;
    }

    result = zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_profiling_data",
                                                 reinterpret_cast<void**>(&_graph_profiling_ddi_table_ext));
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeDriverGetExtensionFunctionAddress failed {0:X+}", uint64_t(result));
        return;
    }

    uint32_t device_count = 1;
    // Get our target device
    result = zeDeviceGet(driver_handle, &device_count, &device_handle);
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeDeviceGet failed {0:X+}", uint64_t(result));
        return;
    }

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    result = zeContextCreate(driver_handle, &context_desc, &context);
    if (ZE_RESULT_SUCCESS != result) {
        log.warning("zeContextCreate failed {0:X+}", uint64_t(result));
        return;
    }

    auto device = std::make_shared<ZeroDevice>(driver_handle, device_handle, context, _graph_ddi_table_ext,
                                               _graph_profiling_ddi_table_ext);
    devices.emplace(std::make_pair(device->getName(), device));
}

ZeroStructsInitializer::~ZeroStructsInitializer() {
    if (context) {
        auto result = zeContextDestroy(context);
        if (ZE_RESULT_SUCCESS != result) {
            log.warning("zeContextDestroy failed {0:X+}", uint64_t(result));
        }
    }
};

ZeroEngineBackend::ZeroEngineBackend(): instance(std::make_unique<ZeroStructsInitializer>()) {
}

ZeroEngineBackend::~ZeroEngineBackend() = default;

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice() const {
    if (instance->getInstanceDevices().size())
        return instance->getInstanceDevices().begin()->second;
    else
        return {};
}

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice(const std::string& /*name*/) const {
    // TODO Add the search of the device by platform & slice
    return getDevice();
}

const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    const auto& devices = instance->getInstanceDevices();
    std::vector<std::string> devicesNames;
    std::for_each(devices.cbegin(), devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });

    return devicesNames;
}

}  // namespace vpux

INFERENCE_PLUGIN_API(void) CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& obj) {
    obj = std::make_shared<vpux::ZeroEngineBackend>();
}
