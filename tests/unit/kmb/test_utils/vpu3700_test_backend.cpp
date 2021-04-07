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

/**
 * @brief These are a set of classes which emulates behavior of a backend with a single device. Provided only for unit tests purposes.
 */
namespace vpux {
 
class DummyVPU3700Device final : public IDevice {
public:
    DummyVPU3700Device() = default;
    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr&/*networkDescription*/, const VPUXConfig& /*config*/) override {
        return nullptr;
    }

    std::shared_ptr<Allocator> getAllocator() const override {
        return nullptr;
    }
    std::string getName() const override {
        return "DummyVPU3700Device";
    }
};

class VPU3700TestBackend final : public vpux::IEngineBackend {
public:
    VPU3700TestBackend() : _dummyDevice(std::make_shared<DummyVPU3700Device>()) { } 

    const std::string getName() const override {
        return "VPU3700TestBackend";
    }

    std::unordered_set<std::string> getSupportedOptions() const override { 
        return {};
    }
    const std::shared_ptr<IDevice> getDevice() const override {
        return _dummyDevice;
    }
    const std::shared_ptr<IDevice> getDevice(const std::string& /*deviceId*/) const override {
        return _dummyDevice;
    }

    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& /*map*/) const override {
        return _dummyDevice;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {_dummyDevice->getName()};
    }
private:
    std::shared_ptr<DummyVPU3700Device> _dummyDevice;
};

} // namespace vpux

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new vpux::VPU3700TestBackend();
        return InferenceEngine::StatusCode::OK;
    } catch (const std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
