//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux.hpp>
#include <vpux/vpux_plugin_params.hpp>

/**
 * @brief These are a set of classes which emulates behavior of a backend with a single device. Provided only for unit
 * tests purposes.
 */
namespace vpux {

class DummyVPU3700Device final : public IDevice {
public:
    DummyVPU3700Device() {
    }
    std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr /*networkDescription*/,
                                             const Config& /*config*/) override {
        return nullptr;
    }

    std::string getName() const override {
        return "DummyVPU3700Device";
    }
    std::string getFullDeviceName() const override {
        return "Intel(R) NPU (DummyVPU3700Device)";
    }

    std::string dummyGetDeviceId() const {
        return "3700";
    }

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> /*compiledModel*/,
            const std::shared_ptr<const NetworkDescription> /*networkDescription*/, const Executor::Ptr /*executor*/,
            const Config& /*config*/) override {
        return nullptr;
    }
};

class VPU3700TestBackend final : public vpux::IEngineBackend {
public:
    VPU3700TestBackend(): _dummyDevice(std::make_shared<DummyVPU3700Device>()) {
    }

    const std::string getName() const override {
        return "VPU3700TestBackend";
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

    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& paramMap) const override {
        const auto pm = paramMap.find(InferenceEngine::VPUX_PARAM_KEY(DEVICE_ID));
        std::string deviceId = pm->second.as<std::string>();

        if (deviceId == _dummyDevice->dummyGetDeviceId())
            return _dummyDevice;
        else
            return nullptr;
    }

    const std::vector<std::string> getDeviceNames() const override {
        return {_dummyDevice->getName(), "noOtherDevice"};
    }

private:
    std::shared_ptr<DummyVPU3700Device> _dummyDevice;
};

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend, const vpux::Config&) {
    backend = std::make_shared<vpux::VPU3700TestBackend>();
}
