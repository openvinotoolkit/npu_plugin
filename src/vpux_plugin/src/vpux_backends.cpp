//
// Copyright 2020-2021 Intel Corporation.
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

// Plugin
#include "vpux_backends.h"

#include <fstream>
#include <memory>

#include "vpux_exceptions.h"
#include "vpux_remote_context.h"

#include <device_helpers.hpp>

namespace vpux {
namespace IE = InferenceEngine;

// TODO Config will be useless here, since only default values will be used
VPUXBackends::VPUXBackends(const std::vector<std::string>& backendRegistry)
        : _logger(vpu::Logger("VPUXBackends", vpu::LogLevel::Error, vpu::consoleOutput())) {
    std::vector<std::shared_ptr<EngineBackend>> registeredBackends;
    for (const auto& name : backendRegistry) {
        const auto path = getLibFilePath(name);
        const auto exists = std::ifstream(path.c_str()).good();
        if (exists) {
            try {
                const auto backend = std::make_shared<EngineBackend>(path);
                if (backend->getDeviceNames().size() != 0) {
                    _logger.debug("Register %s", name);
                    registeredBackends.emplace_back(backend);
                }
            } catch (const IE::Exception& e) {
                _logger.warning("Exception '%s' while searching for a device by %s", e.what(), name);
            } catch (...) {
                _logger.warning("Unknown exception while searching for a device by %s", name);
            }
        }
    }
    if (registeredBackends.empty()) {
        _logger.warning("Cannot find backend for inference. Make sure if device is available.");
        registeredBackends.emplace_back(nullptr);
    }
    // TODO: implementation of getDevice methods needs to be updated to go over all
    // registered backends to search a device.
    // A single backend is chosen for now to keep existing behavior
    _backend = *registeredBackends.begin();
}

std::string VPUXBackends::getBackendName() const {
    if (_backend != nullptr) {
        return _backend->getName();
    }

    return "";
}

std::shared_ptr<Device> VPUXBackends::getDevice(const std::string& specificName) const {
    _logger.debug("Searching for device %s to use started...", specificName);
    // TODO iterate over all available backends
    std::shared_ptr<Device> deviceToUse = nullptr;

    if (_backend != nullptr) {
        if (specificName.empty()) {
            deviceToUse = _backend->getDevice();
        } else {
            deviceToUse = _backend->getDevice(specificName);
        }
    }

    if (deviceToUse == nullptr) {
        _logger.warning("Device to use not found!");
    } else {
        _logger.debug("Device to use found: %s", deviceToUse->getName());
    }
    return deviceToUse;
}

std::shared_ptr<Device> VPUXBackends::getDevice(const IE::ParamMap& paramMap) const {
    return _backend->getDevice(paramMap);
}

std::shared_ptr<Device> VPUXBackends::getDevice(const IE::RemoteContext::Ptr& context) const {
    // TODO more complicated logic should be here. Might require changing in backend implementation
    const auto privateContext = std::dynamic_pointer_cast<VPUXRemoteContext>(context);
    if (context == nullptr) {
        IE_THROW() << FAILED_CAST_CONTEXT;
    }
    const auto device = privateContext->getDevice();
    _logger.debug("Device from context found: {}", device->getName());
    return device;
}

std::vector<std::string> VPUXBackends::getAvailableDevicesNames() const {
    return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
}

std::unordered_set<std::string> VPUXBackends::getSupportedOptions() const {
    return _backend == nullptr ? std::unordered_set<std::string>() : _backend->getSupportedOptions();
}

// TODO config should be also specified to backends, to allow use logging in devices and all levels below
void VPUXBackends::setup(const VPUXConfig& config) {
    _logger.setLevel(config.logLevel());
}

static std::map<IE::VPUXConfigParams::VPUXPlatform, std::string> compilationPlatformMap = {
        {IE::VPUXConfigParams::VPUXPlatform::VPU3400_A0, "3400_A0"},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3400, "3700"},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3700, "3700"},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3800, "3900"},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3900, "3900"},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},
};

std::string VPUXBackends::getCompilationPlatform(const IE::VPUXConfigParams::VPUXPlatform platform,
                                                 const std::string& deviceId) const {
    // Platform parameter has a higher priority than deviceID
    if (platform != IE::VPUXConfigParams::VPUXPlatform::AUTO) {
        return compilationPlatformMap.at(platform);
    }

    // Get compilation platform from deviceID
    if (!deviceId.empty()) {
        return utils::getPlatformNameByDeviceName(deviceId);
    }

    // Automatic detection of compilation platform
    const auto devNames = getAvailableDevicesNames();
    if (devNames.empty()) {
        IE_THROW() << "No devices found - DEVICE_ID with platform is required for compilation";
    }

    const auto compilationPlatform = utils::getPlatformByDeviceName(devNames.at(0));
    const auto compilationPlatformName = compilationPlatformMap.at(compilationPlatform);
    const auto anotherPlatformIt =
            std::find_if(devNames.cbegin(), devNames.cend(), [&compilationPlatformName](const std::string& devName) {
                const auto curCompilationPlatform = utils::getPlatformByDeviceName(devName);
                const auto curCompilationPlatformName = compilationPlatformMap.at(curCompilationPlatform);
                return (curCompilationPlatformName != compilationPlatformName);
            });
    if (anotherPlatformIt != devNames.cend()) {
        IE_THROW() << "Different VPUX platform have been detected. Not supported configuration.";
    }

    return compilationPlatformName;
}

}  // namespace vpux
