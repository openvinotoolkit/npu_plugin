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

#include "hddl2_exceptions.h"
// Subplugin - TODO remove this dependencies
#include "hddl2_remote_context.h"
#include "subplugin/hddl2_backend.h"

namespace vpux {
namespace IE = InferenceEngine;

// TODO Config will be useless here, since only default values will be used
vpux::VPUXBackends::VPUXBackends(const VPUXConfig& config)
    : _logger(std::make_shared<vpu::Logger>("VPUXBackends", config.logLevel(), vpu::consoleOutput())) {
    // TODO Engine should have ability to work with different backends and scope of one plugin object
    {
        // TODO Hardcoded for now. Move to separate lib and use vpux::EngineBackendConfigurator::findBackend();
        // TODO default config for now.
        std::shared_ptr<vpux::IEngineBackend> backEnd = std::make_shared<vpux::HDDL2::HDDL2Backend>(config);
        // TODO Print name of backend. Required getName interface for vpux::Backend class
        _logger->debug("Backend found!");
        _backends.emplace_back(backEnd);
    }
}

std::shared_ptr<vpux::IDevice> VPUXBackends::getDeviceToUse(const std::string& specificDeviceName) const {
    _logger->debug("Searching for device to use started...");
    // TODO iterate over all available backends
    std::shared_ptr<vpux::IDevice> deviceToUse = nullptr;
    for (const auto& backend : _backends) {
        // TODO Ignore default VPU-0. Track #S-38444
        const std::string ignoredDeviceName("VPU-0");
        if (specificDeviceName.empty() || specificDeviceName == ignoredDeviceName) {
            const auto& devices = backend->getDevices();
            // Get first available device and exit
            if (!devices.empty()) {
                deviceToUse = devices.begin()->second;
                break;
            }
        } else {
            // TODO Implementation for HDDL2Backend specific. Required investigation for real solution
            auto privateBackend = std::dynamic_pointer_cast<HDDL2::HDDL2Backend>(backend);
            if (privateBackend == nullptr) {
                THROW_IE_EXCEPTION << "Get getAvailableDevicesNames not implemented for this backend";
            }
            deviceToUse = privateBackend->getDevice(specificDeviceName);
        }
    }
    if (deviceToUse == nullptr) {
        _logger->warning("Device to use not found!");
    } else {
        _logger->debug("Device to use found: %s", deviceToUse->getName());
    }
    return deviceToUse;
}

std::shared_ptr<vpux::IDevice> VPUXBackends::getDeviceFromContext(
    const InferenceEngine::RemoteContext::Ptr& context) const {
    // TODO more complicated logic should be here. Might require changing in backend implementation
    auto privateContext = std::dynamic_pointer_cast<vpu::HDDL2Plugin::HDDL2RemoteContext>(context);
    if (privateContext == nullptr) {
        THROW_IE_EXCEPTION << "Get getDeviceFromContext not implemented for this backend";
    }
    auto device = privateContext->getDevice();
    _logger->debug("Device from context found: {}", device->getName());
    return device;
}

std::vector<std::string> VPUXBackends::getAvailableDevicesNames() const {
    std::vector<std::string> deviceNames;
    // TODO device name should have prefix for each backend
    for (const auto& backend : _backends) {
        // TODO currently implemented only for HDDL2Backend. Required changes in vpux backend interface
        auto privateBackend = std::dynamic_pointer_cast<HDDL2::HDDL2Backend>(backend);
        if (privateBackend == nullptr) {
            THROW_IE_EXCEPTION << "Get getAvailableDevicesNames not implemented for this backend";
        }
        const auto devices = privateBackend->getDevicesNames();
        deviceNames.insert(deviceNames.begin(), devices.cbegin(), devices.cend());
    }
    std::sort(deviceNames.begin(), deviceNames.end());
    return deviceNames;
}
void VPUXBackends::setup(const VPUXConfig& config) const { _logger->setLevel(config.logLevel()); }

}  // namespace vpux
