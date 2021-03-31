//
// Copyright 2020 Intel Corporation.
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
