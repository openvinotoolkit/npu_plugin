//
// Copyright 2021 Intel Corporation.
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

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new vpux::ThrowTestBackend();
        return InferenceEngine::StatusCode::OK;
    } catch (const std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
