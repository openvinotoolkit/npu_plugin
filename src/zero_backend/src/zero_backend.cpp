//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_backend.h"

#include <description_buffer.hpp>
#include <vector>

#include "ze_intel_vpu_uuid.h"
#include "zero_device.h"
#include "zero_utils.h"

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

const ze_driver_uuid_t ZeroStructsInitializer::uuid = ze_intel_vpu_driver_uuid;

ZeroStructsInitializer::ZeroStructsInitializer(): log(Logger::global().nest("ZeroStructsInitializer", 0)) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroStructsInitializer::ZeroStructsInitializer");
    zeroUtils::throwOnFail("zeInit", zeInit(ZE_INIT_FLAG_VPU_ONLY));

    ze_driver_handle_t driver_handle = nullptr;
    ze_device_handle_t device_handle = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    uint32_t drivers = 0;
    zeroUtils::throwOnFail("zeDriverGet", zeDriverGet(&drivers, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(drivers);
    zeroUtils::throwOnFail("zeDriverGet", zeDriverGet(&drivers, all_drivers.data()));

    // Get our target driver
    ze_driver_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    for (uint32_t i = 0; i < drivers; ++i) {
        zeDriverGetProperties(all_drivers[i], &props);

        if (memcmp(&props.uuid, &uuid, sizeof(uuid)) == 0) {
            driver_handle = all_drivers[i];
            break;
        }
    }
    if (driver_handle == nullptr) {
        IE_THROW() << "zeDriverGet failed to return VPU driver";
    }

    // Check L0 API version
    ze_api_version_t ze_drv_api_version = {};
    zeroUtils::throwOnFail("zeDriverGetApiVersion", zeDriverGetApiVersion(driver_handle, &ze_drv_api_version));

    if (ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MAJOR_VERSION(ze_drv_api_version)) {
        IE_THROW() << "Incompatibility between VPU plugin and driver! "
                   << "Plugin L0 API major version = " << ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT) << ", "
                   << "Driver L0 API major version = " << ZE_MAJOR_VERSION(ze_drv_api_version);
    }
    if (ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MINOR_VERSION(ze_drv_api_version)) {
        log.warning("Some features might not be available! "
                    "Plugin L0 API minor version = {0}, Driver L0 API minor version = {1}",
                    ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT), ZE_MINOR_VERSION(ze_drv_api_version));
    }

    // Load our graph extension
    zeroUtils::throwOnFail("zeDriverGetExtensionFunctionAddress",
                           zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_graph",
                                                               reinterpret_cast<void**>(&_graph_ddi_table_ext)));

    zeroUtils::throwOnFail(
            "zeDriverGetExtensionFunctionAddress",
            zeDriverGetExtensionFunctionAddress(driver_handle, "ZE_extension_profiling_data",
                                                reinterpret_cast<void**>(&_graph_profiling_ddi_table_ext)));

    uint32_t device_count = 1;
    // Get our target device
    zeroUtils::throwOnFail("zeDeviceGet", zeDeviceGet(driver_handle, &device_count, &device_handle));

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    zeroUtils::throwOnFail("zeContextCreate", zeContextCreate(driver_handle, &context_desc, &context));

    auto device = std::make_shared<ZeroDevice>(driver_handle, device_handle, context, _graph_ddi_table_ext,
                                               _graph_profiling_ddi_table_ext);
    devices.emplace(std::make_pair(device->getName(), device));
}

ZeroStructsInitializer::~ZeroStructsInitializer() {
    if (context) {
        auto result = zeContextDestroy(context);
        if (ZE_RESULT_SUCCESS != result) {
            log.error("zeContextDestroy failed {0:X+}", uint64_t(result));
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
