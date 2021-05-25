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
 * @brief This is a class which emulates behavior of a backend which throws exceptions. Provided only for unit tests purposes.
 */
class ThrowTestBackend final : public vpux::IEngineBackend {
public:
    ThrowTestBackend() {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
    }

    const std::string getName() const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return "ThrowTest";
    }

    std::unordered_set<std::string> getSupportedOptions() const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return {};
    }
    const std::shared_ptr<IDevice> getDevice() const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return nullptr;
    }
    const std::shared_ptr<IDevice> getDevice(const std::string& /*deviceId*/) const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& /*map*/) const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
        return {};
    }
};

} // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::ThrowTestBackend>();
}
