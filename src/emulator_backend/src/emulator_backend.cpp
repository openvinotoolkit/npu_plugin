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

#include "emulator_backend.hpp"

#include <ie_common.h>
#include <description_buffer.hpp>
#include <device_helpers.hpp>

#include "emulator_device.hpp"

namespace ie = InferenceEngine;

namespace vpux {

EmulatorBackend::EmulatorBackend()
        : _logger(std::unique_ptr<vpu::Logger>(
                  // [Track number: S#42840]
                  // TODO: config will come by another PR, for now let's use Error log level [Track number: S#42840]
                  new vpu::Logger("EmulatorBackend", vpu::LogLevel::Error /*_config.logLevel()*/,
                                  vpu::consoleOutput()))),
          _device(std::make_shared<EmulatorDevice>()) {
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
    if (utils::getPlatformByDeviceName(name) == ie::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        return _device;
    }

    return nullptr;
}
const std::vector<std::string> EmulatorBackend::getDeviceNames() const {
    std::vector<std::string> availableDevices = {_device->getName()};
    return availableDevices;
}

}  // namespace vpux

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new vpux::EmulatorBackend();
        return ie::StatusCode::OK;
    } catch (std::exception& ex) {
        return ie::DescriptionBuffer(ie::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
