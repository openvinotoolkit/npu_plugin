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

// System
#include <fstream>
#include <memory>
// Plugin
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
          _logger(std::make_shared<vpu::Logger>("HDDL2Backend", vpu::LogLevel::Error /*_config.logLevel()*/,
                                                vpu::consoleOutput())) {
    setUniteLogLevel(vpu::LogLevel::Error /*_config.logLevel()*/);
    _devices = createDeviceMap();
}

/** Generic device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice() const {
    return getDeviceNames().empty() ? nullptr : std::make_shared<ImageWorkloadDevice>();
}

/** Specific device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const std::string& specificDeviceName) const {
    const auto devices = getDeviceNames();
    const auto it = std::find(devices.cbegin(), devices.cend(), specificDeviceName);
    if (it != devices.end()) {
        return std::make_shared<ImageWorkloadDevice>(*it);
    } else {
        // TODO Old naming format VPUX.swDeviceId is deprecated, need to be removed in the future
        // Check old naming format (deprecated)
        uint32_t swDeviceId;
        try {
            swDeviceId = static_cast<uint32_t>(std::stol(specificDeviceName));
        } catch (...) {
            return nullptr;
        }
        const auto devMap = getSwDeviceIdNameMap();
        const auto devIt = devMap.find(swDeviceId);
        if (devIt != devMap.end()) {
            return std::make_shared<ImageWorkloadDevice>((*devIt).second);
        }

        return nullptr;
    }
}

const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const InferenceEngine::ParamMap& paramMap) const {
    return std::make_shared<VideoWorkloadDevice>(paramMap);
}

const std::vector<std::string> HDDL2Backend::getDeviceNames() const {
    // TODO: [Track number: S#42053]
    if (!isServiceAvailable() || !isServiceRunning()) {
        // return empty device list if service is not available or service is not running
        _logger->warning("HDDL2 service is not available or service is not running!");
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
        devices.insert({"HDDL2", std::make_shared<ImageWorkloadDevice>()});
        _logger->debug("HDDL2 devices found for execution.");
    } else {
        _logger->debug("HDDL2 devices not found for execution.");
    }
    return devices;
}

bool HDDL2Backend::isServiceAvailable(const vpu::Logger::Ptr& logger) {
    const std::ifstream defaultService("/opt/intel/hddlunite/bin/hddl_scheduler_service");

    const std::string specifiedServicePath =
            std::getenv("KMB_INSTALL_DIR") != nullptr ? std::getenv("KMB_INSTALL_DIR") : "";
    const std::ifstream specifiedService(specifiedServicePath + std::string("/bin/hddl_scheduler_service"));
    const std::ifstream specifiedCustomService(specifiedServicePath + std::string("/hddl_scheduler_service"));

    const auto serviceAvailable =
            specifiedService.good() || specifiedCustomService.good() || defaultService.good() || isServiceRunning();

    if (logger) {
        serviceAvailable ? logger->debug(SERVICE_AVAILABLE.c_str()) : logger->debug(SERVICE_NOT_AVAILABLE.c_str());
    }
    return serviceAvailable;
}

bool HDDL2Backend::isServiceRunning() {
    return HddlUnite::isServiceRunning();
}

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<HDDL2Backend>();
}

}  // namespace hddl2
}  // namespace vpux
