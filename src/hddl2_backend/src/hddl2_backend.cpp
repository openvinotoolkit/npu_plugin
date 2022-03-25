//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

// System
#include <fstream>
#include <memory>
// Plugin
#include <device_helpers.hpp>
#include "hddl2_exceptions.h"
// Subplugin
#include "hddl2_backend.h"
#include "hddl2_helper.h"
#include "image_workload_device.h"
#include "video_workload_device.h"
// Low-level
#include <HddlUnite.h>

namespace vpux {
namespace hddl2 {

HDDL2Backend::HDDL2Backend()
        :  // [Track number: S#42840]
           // TODO: config will come by another PR, for now let's use Error log level
          _logger("HDDL2Backend", LogLevel::Error /*_config.logLevel()*/) {
    setUniteLogLevel(_logger);
    _devices = createDeviceMap();
}

/** Generic device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice() const {
    return getDeviceNames().empty() ? nullptr : std::make_shared<ImageWorkloadDevice>();
}

/** Specific device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const std::string& specificDeviceName) const {
    // Search for "platform.slice_id" naming format
    const auto devices = getDeviceNames();
    const auto it = std::find(devices.cbegin(), devices.cend(), specificDeviceName);
    if (it != devices.end()) {
        return std::make_shared<ImageWorkloadDevice>(*it);
    }

    // Search for "only platform" naming format
    const auto expectedPlatformName = utils::getPlatformNameByDeviceName(specificDeviceName);
    for (const auto& curDev : devices) {
        const auto currentPlatformName = utils::getPlatformNameByDeviceName(curDev);
        if (currentPlatformName == expectedPlatformName) {
            return std::make_shared<ImageWorkloadDevice>(curDev);
        }
    }

    return nullptr;
}

const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const InferenceEngine::ParamMap& paramMap) const {
    return std::make_shared<VideoWorkloadDevice>(paramMap, _logger.level());
}

const std::vector<std::string> HDDL2Backend::getDeviceNames() const {
    // TODO: [Track number: S#42053]
    if (!isServiceAvailable() || !isServiceRunning()) {
        // return empty device list if service is not available or service is not running
        _logger.warning("HDDL2 service is not available or service is not running!");
        return std::vector<std::string>();
    }

    std::vector<std::string> devicesNames;
    for (const auto& dev : getSwDeviceIdNameMap()) {
        devicesNames.push_back(dev.second);
    }
    std::sort(devicesNames.begin(), devicesNames.end());
    return devicesNames;
}

std::map<std::string, std::shared_ptr<vpux::IDevice>> HDDL2Backend::createDeviceMap() {
    std::map<std::string, std::shared_ptr<IDevice>> devices;
    // TODO Add more logs and cases handling
    if (isServiceAvailable(_logger) && !getDeviceNames().empty()) {
        devices.insert({"AUTO", std::make_shared<ImageWorkloadDevice>()});
        _logger.debug("HDDL2 devices found for execution.");
    } else {
        _logger.debug("HDDL2 devices not found for execution.");
    }
    return devices;
}

bool HDDL2Backend::isServiceAvailable(Logger logger) {
    const std::ifstream defaultService("/opt/intel/hddlunite/bin/hddl_scheduler_service");

    const std::string specifiedServicePath =
            std::getenv("KMB_INSTALL_DIR") != nullptr ? std::getenv("KMB_INSTALL_DIR") : "";
    const std::ifstream specifiedService(specifiedServicePath + std::string("/bin/hddl_scheduler_service"));
    const std::ifstream specifiedCustomService(specifiedServicePath + std::string("/hddl_scheduler_service"));

    const auto serviceAvailable =
            specifiedService.good() || specifiedCustomService.good() || defaultService.good() || isServiceRunning();

    logger.debug("{0}", serviceAvailable ? SERVICE_AVAILABLE : SERVICE_NOT_AVAILABLE);
    return serviceAvailable;
}

bool HDDL2Backend::isServiceRunning() {
    return HddlUnite::isServiceRunning();
}

INFERENCE_PLUGIN_API(void) CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& obj) {
    obj = std::make_shared<HDDL2Backend>();
}

}  // namespace hddl2
}  // namespace vpux
