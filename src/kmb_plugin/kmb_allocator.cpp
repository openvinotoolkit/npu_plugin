//
// Copyright 2019 Intel Corporation.
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

#include "kmb_allocator.h"

#include <memory>
#include <string>

#include "kmb_native_allocator.h"
#include "kmb_udma_allocator.h"
#include "kmb_vpusmm_allocator.h"
#include "ie_macro.hpp"

using namespace vpu::KmbPlugin;

void* KmbAllocator::lock(void* handle, InferenceEngine::LockOp) noexcept {
    if (_allocatedMemory.find(handle) == _allocatedMemory.end()) return nullptr;

    return handle;
}

void KmbAllocator::unlock(void* handle) noexcept { UNUSED(handle); } //cpplint mark this line as false positive

unsigned long KmbAllocator::getPhysicalAddress(void* handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return 0;
    }

    auto memoryDesc = memoryIt->second;
    return memoryDesc.physAddr;
}

bool KmbAllocator::isValidPtr(void* ptr) noexcept { return ptr != nullptr; }

std::shared_ptr<KmbAllocator>& vpu::KmbPlugin::getKmbAllocator() {
    static std::shared_ptr<KmbAllocator> allocator;
    if (allocator == nullptr) {
        const char* allocatorEnvPtr = std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE");
        std::string allocatorType;
        if (allocatorEnvPtr) {
            allocatorType = allocatorEnvPtr;
        }
        if (allocatorType == "UDMA") {
            allocator = std::make_shared<KmbUdmaAllocator>();
        } else if (allocatorType == "NATIVE") {
            allocator = std::make_shared<KmbNativeAllocator>();
        } else {
            allocator = std::make_shared<KmbVpusmmAllocator>();
        }
    }
    return allocator;
}

void* KmbAllocator::wrapRemoteMemory(const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, void* memHandle) noexcept {
    void* virtAddr = mmap(memHandle, size, PROT_READ | PROT_WRITE, MAP_SHARED, remoteMemoryFd, 0);

    if (virtAddr == MAP_FAILED) return nullptr;

    return virtAddr;
}
