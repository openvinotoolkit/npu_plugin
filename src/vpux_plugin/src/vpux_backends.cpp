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

// Plugin
#include "vpux_backends.h"

#include "vpux_exceptions.h"
#include "vpux_remote_context.h"

namespace vpux {
namespace IE = InferenceEngine;

// TODO Config will be useless here, since only default values will be used
VPUXBackends::VPUXBackends(const VPUXConfig& config)
        : _logger(std::make_shared<vpu::Logger>("VPUXBackends", config.logLevel(), vpu::consoleOutput())),
          _backend(EngineBackendConfigurator::findBackend({{CONFIG_KEY(LOG_LEVEL), config.logLevel()}})) {
}

std::shared_ptr<Device> VPUXBackends::getDevice(const std::string& specificName) const {
    _logger->debug("Searching for device to use started...");
    // TODO iterate over all available backends
    std::shared_ptr<Device> deviceToUse = nullptr;
    // TODO Ignore default VPU-0. Track #S-38444
    const std::string ignoredDeviceName("VPU-0");

    if (specificName.empty() || specificName == ignoredDeviceName) {
        if (_backend != nullptr) {
            deviceToUse = _backend->getDevice();
        }
    } else {
        deviceToUse = _backend->getDevice(specificName);
    }

    if (deviceToUse == nullptr) {
        _logger->warning("Device to use not found!");
    } else {
        _logger->debug("Device to use found: %s", deviceToUse->getName());
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
    _logger->debug("Device from context found: {}", device->getName());
    return device;
}

std::vector<std::string> VPUXBackends::getAvailableDevicesNames() const {
    return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
}

std::unordered_set<std::string> VPUXBackends::getSupportedOptions() const {
    return _backend == nullptr ? std::unordered_set<std::string>() : _backend->getSupportedOptions();
}

// TODO config should be also specified to backends, to allow use logging in devices and all levels below
void VPUXBackends::setup(const VPUXConfig& config) const {
    _logger->setLevel(config.logLevel());
}

}  // namespace vpux
