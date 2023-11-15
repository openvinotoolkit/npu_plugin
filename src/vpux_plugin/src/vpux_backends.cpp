//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_backends.h"

#include <fstream>
#include <memory>

#include "vpux/al/config/common.hpp"
#include "vpux_exceptions.h"

#include <device_helpers.hpp>

// TODO: E#30339 the creation of backend is not scalable,
// it needs to be refactored in order to simplify
// adding other backends into static build config
#ifndef OPENVINO_STATIC_LIBRARY
#include "vpux/utils/core/library_path.hpp"
#elif defined(ENABLE_ZEROAPI_BACKEND)
#include <zero_backend.h>
#endif

namespace vpux {
namespace ie = InferenceEngine;

// TODO Config will be useless here, since only default values will be used
VPUXBackends::VPUXBackends(const std::vector<std::string>& backendRegistry, const Config& config)
        : _logger(Logger::global().nest("NPUBackends", 0)) {
    std::vector<std::shared_ptr<EngineBackend>> registeredBackends;
    const auto registerBackend = [&](std::shared_ptr<EngineBackend> backend, const std::string& name) {
        const auto backendDevices = backend->getDeviceNames();
        if (!backendDevices.empty()) {
            _logger.debug("Register '{0}' with devices '{1}'", name, backendDevices);
            registeredBackends.emplace_back(backend);
        }
    };

#ifndef OPENVINO_STATIC_LIBRARY
    for (const auto& name : backendRegistry) {
        _logger.debug("Try '{0}' backend", name);

        const auto path = getLibFilePath(name);

        const auto exists = std::ifstream(path).good();
        if (!exists) {
            _logger.debug("Backend '{0}' at '{1}' doesn't exist", name, path);
            continue;
        }

        try {
            const auto backend = std::make_shared<EngineBackend>(path, config);
            registerBackend(backend, name);
        } catch (const std::exception& ex) {
            _logger.error("Got an error during backend '{0}' loading : {1}", name, ex.what());
        } catch (...) {
            _logger.error("Got an unknown error during backend '{0}' loading", name);
        }
    }
#else

    (void)backendRegistry;
    (void)config;
#ifdef ENABLE_ZEROAPI_BACKEND
    const auto backend = std::make_shared<EngineBackend>(std::make_shared<ZeroEngineBackend>(config));
#else
    const auto backend = std::make_shared<EngineBackend>(nullptr);
    IE_THROW() << "No backends available. The only available backend for static library configuration is "
                  "npu_level_zero_backend."
               << "Please make sure that ENABLE_ZEROAPI_BACKEND is ON";
#endif
    registerBackend(backend, "npu_level_zero_backend");

#endif

    if (registeredBackends.empty()) {
        registeredBackends.emplace_back(nullptr);
    }

    // TODO: implementation of getDevice methods needs to be updated to go over all
    // registered backends to search a device.
    // A single backend is chosen for now to keep existing behavior
    _backend = *registeredBackends.begin();

    if (_backend != nullptr) {
        _logger.info("Use '{0}' backend for inference", _backend->getName());
    } else {
        _logger.error("Cannot find backend for inference. Make sure the device is available.");
    }
}

std::string VPUXBackends::getBackendName() const {
    if (_backend != nullptr) {
        return _backend->getName();
    }

    return "";
}

std::shared_ptr<Device> VPUXBackends::getDevice(const std::string& specificName) const {
    _logger.debug("Searching for device {0} to use started...", specificName);
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
        _logger.warning("Device not found!");
    } else {
        _logger.debug("Device found: {0}", deviceToUse->getName());
    }
    return deviceToUse;
}

std::shared_ptr<Device> VPUXBackends::getDevice(const ie::ParamMap& paramMap) const {
    return _backend->getDevice(paramMap);
}

std::vector<std::string> VPUXBackends::getAvailableDevicesNames() const {
    return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
}

void VPUXBackends::registerOptions(OptionsDesc& options) const {
    if (_backend != nullptr) {
        _backend->registerOptions(options);
    }
}

// TODO config should be also specified to backends, to allow use logging in devices and all levels below
void VPUXBackends::setup(const Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
}

static std::map<ie::VPUXConfigParams::VPUXPlatform, std::string> compilationPlatformMap = {
        {ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "AUTO_DETECT"},
        {ie::VPUXConfigParams::VPUXPlatform::VPU3700, "3700"},
        {ie::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},
};

std::string VPUXBackends::getCompilationPlatform(const ie::VPUXConfigParams::VPUXPlatform platform,
                                                 const std::string& deviceId) const {
    // Platform parameter has a higher priority than deviceID
    if (platform != ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT &&
        platform != ie::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        return compilationPlatformMap.at(platform);
    }

    // Get compilation platform from deviceID
    if (!deviceId.empty()) {
        return utils::getPlatformNameByDeviceName(deviceId);
    }

    // Automatic detection of compilation platform
    const auto devNames = getAvailableDevicesNames();
    if (devNames.empty()) {
        IE_THROW() << "No devices found - platform must be explicitly specified for compilation. Example: -d NPU.3700 "
                      "instead of -d NPU.";
    }

    if (std::find(devNames.cbegin(), devNames.cend(), "EMULATOR") != devNames.end()) {
        IE_THROW() << "Emulator device is available, but was not explicitly requested";
    }

    // check whether the compilation platform for the default platform name exists within defined platforms
    const auto compilationPlatform = utils::getPlatformByDeviceName(devNames.at(0));
    const auto compilationPlatformName = compilationPlatformMap.at(compilationPlatform);
    const auto anotherPlatformIt =
            std::find_if(devNames.cbegin(), devNames.cend(), [&compilationPlatformName](const std::string& devName) {
                const auto curCompilationPlatform = utils::getPlatformByDeviceName(devName);
                const auto curCompilationPlatformName = compilationPlatformMap.at(curCompilationPlatform);
                return (curCompilationPlatformName == compilationPlatformName);
            });
    if (anotherPlatformIt == devNames.cend()) {
        IE_THROW() << "Different VPUX platform have been detected. Not supported configuration.";
    }

    return compilationPlatformName;
}

}  // namespace vpux
