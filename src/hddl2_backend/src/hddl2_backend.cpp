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
#include "hddl2_context_device.h"
#include "hddl2_device.h"
// Low-level
#include <HddlUnite.h>

using namespace vpux::HDDL2;
namespace vpux {
namespace HDDL2 {

HDDL2Backend::HDDL2Backend(const VPUXConfig& config)
    : _logger(std::make_shared<vpu::Logger>("HDDL2Backend", config.logLevel(), vpu::consoleOutput())),
      _devices(createDeviceMap()) {
    if (_devices.empty()) THROW_IE_EXCEPTION << "Device map is empty.";
}

/** Generic device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice() const {
    return getDeviceNames().empty() ? nullptr : std::make_shared<HDDLUniteDevice>();
}

/** Specific device */
const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const std::string& specificDeviceName) const {
    const auto devices = getDeviceNames();
    const auto it = std::find(devices.cbegin(), devices.cend(), specificDeviceName);
    if (it != devices.end()) {
        return std::make_shared<HDDLUniteDevice>(*it);
    } else {
        return nullptr;
    }
}

const std::shared_ptr<IDevice> HDDL2Backend::getDevice(const InferenceEngine::ParamMap& paramMap) const {
    return std::make_shared<HDDLUniteContextDevice>(paramMap);
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
        THROW_IE_EXCEPTION << "Failed to get devices names!";
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
        devices.insert({"HDDL2", std::make_shared<HDDLUniteDevice>()});
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

bool HDDL2Backend::isServiceRunning() { return HddlUnite::isServiceRunning(); }

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new HDDL2Backend();
        return InferenceEngine::StatusCode::OK;
    } catch (std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}

}  // namespace HDDL2
}  // namespace vpux
