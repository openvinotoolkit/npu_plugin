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

#include "emulator_backend.hpp"

#include <ie_common.h>
#include <description_buffer.hpp>
#include <vpux/kmb_params.hpp>

#include "emulator_device.hpp"

namespace ie = InferenceEngine;

namespace vpux {

EmulatorBackend::EmulatorBackend()
        : _logger(std::unique_ptr<vpu::Logger>(
                  // TODO: config will come by another PR, for now let's use Error log level
                  new vpu::Logger("EmulatorBackend", vpu::LogLevel::Error /*_config.logLevel()*/,
                                  vpu::consoleOutput()))),
          _device(std::make_shared<EmulatorDevice>()) {
}

const std::shared_ptr<IDevice> EmulatorBackend::getDevice() const {
    return _device;
}

const std::shared_ptr<IDevice> EmulatorBackend::getDevice(const ie::ParamMap& map) const {
    // FIXME: parse map to find device
    return _device;
}

const std::shared_ptr<IDevice> EmulatorBackend::getDevice(const std::string& name) const {
    const auto deviceName = _device->getName();
    // TODO: better check?
    if (deviceName.find(name) != std::string::npos) {
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
