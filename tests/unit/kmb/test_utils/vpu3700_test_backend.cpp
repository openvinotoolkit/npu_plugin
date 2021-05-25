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

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::VPU3700TestBackend>();
}
