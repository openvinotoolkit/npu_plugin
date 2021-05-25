//
// Copyright 2021 Intel Corporation.
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

    std::unordered_set<std::string> getSupportedOptions() const override {
        return {};
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

} // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::NoDevicesTestBackend>();
}
