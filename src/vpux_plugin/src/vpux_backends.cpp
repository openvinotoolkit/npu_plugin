//
// Copyright 2020-2021 Intel Corporation.
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
            } catch (const IE::details::InferenceEngineException& e) {
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
    // TODO Ignore default VPU-0. Track #S-38444
    const std::string ignoredDeviceName("VPU-0");

    if (_backend != nullptr) {
        if (specificName.empty() || specificName == ignoredDeviceName) {
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
        THROW_IE_EXCEPTION << FAILED_CAST_CONTEXT;
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
        {IE::VPUXConfigParams::VPUXPlatform::VPU3400_A0, VPUX_CONFIG_VALUE(VPU3400_A0)},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3400, VPUX_CONFIG_VALUE(VPU3700)},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3700, VPUX_CONFIG_VALUE(VPU3700)},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3800, VPUX_CONFIG_VALUE(VPU3900)},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3900, VPUX_CONFIG_VALUE(VPU3900)},
        {IE::VPUXConfigParams::VPUXPlatform::VPU3720, VPUX_CONFIG_VALUE(VPU3720)},
};

std::string VPUXBackends::getCompilationPlatform(const IE::VPUXConfigParams::VPUXPlatform platform) const {
    if (platform != IE::VPUXConfigParams::VPUXPlatform::AUTO) {
        return compilationPlatformMap.at(platform);
    }

    const auto devNames = getAvailableDevicesNames();
    if (devNames.size() > 0) {
        const auto compilationPlatform = utils::getPlatformByDeviceName(devNames.at(0));
        const auto compilationPlatformName = compilationPlatformMap.at(compilationPlatform);
        const auto anotherPlatformIt =
                std::find_if(devNames.cbegin(), devNames.cend(), [compilationPlatformName](const std::string& devName) {
                    const auto curCompilationPlatform = utils::getPlatformByDeviceName(devName);
                    const auto curCompilationPlatformName = compilationPlatformMap.at(curCompilationPlatform);
                    return (curCompilationPlatformName != compilationPlatformName);
                });
        if (anotherPlatformIt != devNames.cend()) {
            THROW_IE_EXCEPTION << "Different VPUX platform have been detected. Not supported configuration.";
        }
        return compilationPlatformName;
    }

    return VPUX_CONFIG_VALUE(VPU3700);
}

}  // namespace vpux
