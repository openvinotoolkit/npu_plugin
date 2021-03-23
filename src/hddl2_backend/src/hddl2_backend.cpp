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

// System
#include <fstream>
#include <memory>
// Plugin
#include "hddl2_exceptions.h"
// Subplugin
#include "hddl2_backend.h"
#include "image_workload_device.h"
#include "video_workload_device.h"
// Low-level
#include <HddlUnite.h>

using namespace vpux::HDDL2;
namespace vpux {
namespace HDDL2 {

// TODO Use config from VPUX Plugin, not default. [Track number: S#42840]
HDDL2Backend::HDDL2Backend(const VPUXConfig& config)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("HDDL2Backend", _config.logLevel(), vpu::consoleOutput())) {
    setUniteLogLevel(_config.logLevel());
    _devices = createDeviceMap();
    if (_devices.empty())
        IE_THROW() << "Device map is empty.";
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

    std::vector<HddlUnite::Device> devices;
    auto status = getAvailableDevices(devices);
    if (status != HDDL_OK) {
        IE_THROW() << "Failed to get devices names!";
    }

    std::vector<std::string> devicesNames;
    for (const auto& device : devices) {
        devicesNames.push_back(std::to_string(device.getSwDeviceId()));
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

    const auto serviceAvailable = specifiedService.good() || defaultService.good() || isServiceRunning();
    if (logger) {
        serviceAvailable ? logger->debug(SERVICE_AVAILABLE.c_str()) : logger->debug(SERVICE_NOT_AVAILABLE.c_str());
    }
    return serviceAvailable;
}

bool HDDL2Backend::isServiceRunning() {
    return HddlUnite::isServiceRunning();
}

HddlUnite::clientLogLevel convertIELogLevelToUnite(const vpu::LogLevel ieLogLevel) {
    switch (ieLogLevel) {
    case vpu::LogLevel::None:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case vpu::LogLevel::Fatal:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    case vpu::LogLevel::Error:
        return HddlUnite::clientLogLevel::LOGLEVEL_ERROR;
    case vpu::LogLevel::Warning:
        return HddlUnite::clientLogLevel::LOGLEVEL_WARN;
    case vpu::LogLevel::Info:
        return HddlUnite::clientLogLevel::LOGLEVEL_INFO;
    case vpu::LogLevel::Debug:
        return HddlUnite::clientLogLevel::LOGLEVEL_DEBUG;
    case vpu::LogLevel::Trace:
        return HddlUnite::clientLogLevel::LOGLEVEL_PROCESS;
    default:
        return HddlUnite::clientLogLevel::LOGLEVEL_FATAL;
    }
}

void HDDL2Backend::setUniteLogLevel(const vpu::LogLevel logLevel) {
    const auto status = HddlUnite::setClientLogLevel(convertIELogLevelToUnite(logLevel));
    if (status != HddlStatusCode::HDDL_OK) {
        _logger->warning("Failed to set client log level for HddlUnite");
    }
}

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<HDDL2Backend>();
}

}  // namespace HDDL2
}  // namespace vpux
