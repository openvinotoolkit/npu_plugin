//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux.hpp>

/**
 * @brief These are a set of classes which emulates behavior of a backend with a single device. Provided only for unit
 * tests purposes.
 */
namespace vpux {

class DummyVPU3720Device final : public IDevice {
public:
    DummyVPU3720Device() {
    }
    std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr /*networkDescription*/,
                                             const Config& /*config*/) override {
        return nullptr;
    }

    std::string getName() const override {
        return "3720.dummyDevice";
    }
    std::string getFullDeviceName() const override {
        return "Intel(R) NPU (DummyVPU3720Device)";
    }

    ov::device::UUID getUuid() const override {
        return {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x37, 0x20};
    }

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> /*compiledModel*/,
            const std::shared_ptr<const NetworkDescription> /*networkDescription*/, const Executor::Ptr /*executor*/,
            const Config& /*config*/) override {
        return nullptr;
    }
};

class VPU3720TestBackend final : public vpux::IEngineBackend {
public:
    VPU3720TestBackend(): _dummyDevice(std::make_shared<DummyVPU3720Device>()) {
    }

    const std::string getName() const override {
        return "VPU3720TestBackend";
    }

    const std::shared_ptr<IDevice> getDevice() const override {
        return _dummyDevice;
    }

    const std::shared_ptr<IDevice> getDevice(const std::string& specificName) const override {
        if (specificName == _dummyDevice->getName())
            return _dummyDevice;
        else
            return nullptr;
    }

    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap&) const override {
        return _dummyDevice;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {_dummyDevice->getName()};
    }

private:
    std::shared_ptr<DummyVPU3720Device> _dummyDevice;
};

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend, const vpux::Config&) {
    backend = std::make_shared<vpux::VPU3720TestBackend>();
}
