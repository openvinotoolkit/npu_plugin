//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "emulator_backend.hpp"

#include "device_helpers.hpp"
#include "emulator_device.hpp"

#include <ie_common.h>
#include <description_buffer.hpp>

namespace ie = InferenceEngine;

namespace vpux {

EmulatorBackend::EmulatorBackend(): _device(std::make_shared<EmulatorDevice>()) {
}

// nullptr is returned to make sure that
// the emulator is chosen only when
// it is requested explicitly by a user
const std::shared_ptr<IDevice> EmulatorBackend::getDevice() const {
    return nullptr;
}

const std::shared_ptr<IDevice> EmulatorBackend::getDevice(const ie::ParamMap& map) const {
    // FIXME: parse map to find device
    return _device;
}

const std::shared_ptr<IDevice> EmulatorBackend::getDevice(const std::string& name) const {
    // TODO: better check?
    if (utils::getPlatformByDeviceName(name) == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        return _device;
    }

    return nullptr;
}
const std::vector<std::string> EmulatorBackend::getDeviceNames() const {
    std::vector<std::string> availableDevices = {_device->getName()};
    return availableDevices;
}

}  // namespace vpux

INFERENCE_PLUGIN_API(void) CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& obj) {
    obj = std::make_shared<vpux::EmulatorBackend>();
}
