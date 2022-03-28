//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux.hpp>

#include <description_buffer.hpp>

namespace vpux {

/**
 * @brief This is a class which emulates behavior of a backend which throws exceptions. Provided only for unit tests
 * purposes.
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

    void registerOptions(OptionsDesc&) const override {
        THROW_IE_EXCEPTION << "Error from ThrowTestBackend";
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

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::ThrowTestBackend>();
}
