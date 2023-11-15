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

class DummyAllocator final : public Allocator {
public:
    DummyAllocator() = default;
    ~DummyAllocator() {
        for (auto ptr : _allocPtrs) {
            delete[] static_cast<char*>(ptr.first);
        }
    }

    void* lock(void* handle, InferenceEngine::LockOp) noexcept override {
        return handle;
    }
    void unlock(void*) noexcept override {
    }

    void* alloc(size_t size) noexcept override {
        void* mem = new char[size];
        _allocPtrs.insert({mem, 1u});
        return mem;
    }

    bool free(void* handle) noexcept override {
        const auto memIter = _allocPtrs.find(handle);
        if (memIter != _allocPtrs.end()) {
            memIter->second--;
        }
        if (!memIter->second) {
            _allocPtrs.erase(handle);
            delete[] static_cast<char*>(handle);
        }
        return true;
    }

    void* wrapRemoteMemoryHandle(const int&, const size_t, void*) noexcept override {
        return nullptr;
    }
    void* wrapRemoteMemoryOffset(const int&, const size_t, const size_t&) noexcept override {
        return nullptr;
    }

    void* wrapRemoteMemory(const InferenceEngine::ParamMap&) noexcept override {
        return nullptr;
    };

    unsigned long getPhysicalAddress(void* /*handle*/) noexcept override {
        return 0;
    }

private:
    std::unordered_map<void*, size_t> _allocPtrs;
};

class DummyVPU3720Device final : public IDevice {
public:
    DummyVPU3720Device(): _allocatorPtr(std::make_shared<DummyAllocator>()) {
    }
    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& /*networkDescription*/,
                                             const Config& /*config*/) override {
        return nullptr;
    }

    std::shared_ptr<Allocator> getAllocator() const override {
        return _allocatorPtr;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& /*paramMap*/) const override {
        return _allocatorPtr;
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

    IInferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& /*networkInputs*/,
                                          const InferenceEngine::OutputsDataMap& /*networkOutputs*/,
                                          const Executor::Ptr& /*executor*/, const Config& /*config*/,
                                          const std::string& /*networkName*/,
                                          const std::vector<std::shared_ptr<const ov::Node>>& /*parameters*/,
                                          const std::vector<std::shared_ptr<const ov::Node>>& /*results*/,
                                          const vpux::NetworkIOVector& /* networkStatesInfo */,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& /*allocator*/) override {
        return nullptr;
    }

private:
    std::shared_ptr<Allocator> _allocatorPtr = nullptr;
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
