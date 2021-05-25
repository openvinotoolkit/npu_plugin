//
// Copyright 2020 Intel Corporation.
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

#include "vpux.hpp"

#include <file_utils.h>

#include <cstdlib>
#include <details/ie_so_pointer.hpp>
#include <memory>

namespace vpux {

// expected format VPU-#, where # is device id
int extractIdFromDeviceName(const std::string& name) {
    const size_t expectedSize = 5;
    if (name.size() != expectedSize) {
#ifdef __aarch64__
        IE_THROW() << "Unexpected device name: " << name;
#else
        return -1;
#endif
    }

    return name.at(expectedSize - 1) - '0';
}

bool isBlobAllocatedByAllocator(const InferenceEngine::Blob::Ptr& blob,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    const auto memoryBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    IE_ASSERT(memoryBlob != nullptr);
    auto lockedMemory = memoryBlob->rmap();
    return allocator->lock(lockedMemory.as<void*>());
}

enum class EngineBackendType : uint8_t { VPUAL = 1, HDDL2 = 2, ZeroApi = 3, Emulator = 4 };

//------------------------------------------------------------------------------
EngineBackend::EngineBackend(std::string pathToLib): _impl(pathToLib) {
}

inline const std::shared_ptr<Device> wrapDeviceWithImpl(const std::shared_ptr<IDevice>& device,
                                                        const IEngineBackendPtr backendPtr) {
    if (device == nullptr) {
        return nullptr;
    }
    return std::make_shared<Device>(device, backendPtr);
}
const std::shared_ptr<Device> EngineBackend::getDevice() const {
    return wrapDeviceWithImpl(_impl->getDevice(), _impl);
}

const std::shared_ptr<Device> EngineBackend::getDevice(const std::string& specificDeviceName) const {
    return wrapDeviceWithImpl(_impl->getDevice(specificDeviceName), _impl);
}

const std::shared_ptr<Device> EngineBackend::getDevice(const InferenceEngine::ParamMap& paramMap) const {
    return wrapDeviceWithImpl(_impl->getDevice(paramMap), _impl);
}

const std::shared_ptr<IDevice> IEngineBackend::getDevice() const {
    IE_THROW() << "Default getDevice() not implemented";
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const std::string&) const {
    IE_THROW() << "Specific device search not implemented";
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const InferenceEngine::ParamMap&) const {
    IE_THROW() << "Get device based on params not implemented";
}
const std::vector<std::string> IEngineBackend::getDeviceNames() const {
    IE_THROW() << "Get all device names not implemented";
}

std::unordered_set<std::string> IEngineBackend::getSupportedOptions() const {
    return {};
}

void* Allocator::wrapRemoteMemory(const InferenceEngine::ParamMap&) noexcept {
    std::cerr << "Wrapping remote memory not implemented" << std::endl;
    return nullptr;
}
std::shared_ptr<Allocator> IDevice::getAllocator(const InferenceEngine::ParamMap&) const {
    IE_THROW() << "Not supported";
}

}  // namespace vpux
