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
#include <unordered_map>
#include <vpux_params_private_options.hpp>
#include "vpux/utils/IE/blob.hpp"

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

    void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept override {
        const auto memoryHandle = [&paramMap]() -> void* {
            const auto memIter = paramMap.find(InferenceEngine::VPUX_PARAM_KEY(MEM_HANDLE));
            if (memIter == paramMap.end()) {
                return nullptr;
            }
            return paramMap.at(InferenceEngine::VPUX_PARAM_KEY(MEM_HANDLE)).as<void*>();
        }();
        if (memoryHandle != nullptr) {
            _allocPtrs[memoryHandle]++;
            return memoryHandle;
        }
        const auto blobSize = [&paramMap]() -> size_t {
            const size_t defValue = 100;
            const auto tensorIter = paramMap.find(InferenceEngine::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC));
            if (tensorIter == paramMap.end()) {
                return defValue;
            }
            const auto tensorDesc = paramMap.at(InferenceEngine::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC))
                                            .as<std::shared_ptr<InferenceEngine::TensorDesc>>();
            return vpux::getMemorySize(*tensorDesc).count();
        }();
        return alloc(blobSize);
    };

    unsigned long getPhysicalAddress(void* /*handle*/) noexcept override {
        return 0;
    }

private:
    std::unordered_map<void*, size_t> _allocPtrs;
};

class DummyVPU3700Device final : public IDevice {
public:
    DummyVPU3700Device(): _allocatorPtr(std::make_shared<DummyAllocator>()) {
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
        return "DummyVPU3700Device";
    }

    // TODO: it is a stub for future implementation
    // currently, nullptr is used as a signal to use InferRequestCommon from vpux_al
    InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& /*networkInputs*/,
                                         const InferenceEngine::OutputsDataMap& /*networkOutputs*/,
                                         const Executor::Ptr& /*executor*/, const Config& /*config*/,
                                         const std::string& /*networkName*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*parameters*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*results*/,
                                         const std::shared_ptr<InferenceEngine::IAllocator>& /*allocator*/) override {
        return nullptr;
    }

private:
    std::shared_ptr<Allocator> _allocatorPtr = nullptr;
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

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::VPU3700TestBackend>();
}
