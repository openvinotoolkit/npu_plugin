//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux.hpp>

#include <description_buffer.hpp>

namespace vpux {

/**
 * @brief This is a class which emulates behavior of a backend without a device. Provided only for unit tests purposes.
 */
class NoDevicesTestBackend final : public vpux::IEngineBackend {
public:
    NoDevicesTestBackend() = default;

    const std::string getName() const override {
        return "OneDeviceTestBackend";
    }

    const std::shared_ptr<IDevice> getDevice() const override {
        return nullptr;
    }
    const std::shared_ptr<IDevice> getDevice(const std::string& /*deviceId*/) const override {
        return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& /*map*/) const override {
        return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {};
    }
};

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::NoDevicesTestBackend>();
}
