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

#include "hddl2_remote_allocator.h"

#include <memory>

using namespace vpu::HDDL2Plugin;

bool static isValidInput(size_t size) noexcept {
    if (size <= 0 || size > MAX_ALLOC_SIZE) {
        printf("%s: Incorrect size!\n", __FUNCTION__);
        return false;
    }
    return true;
}

HDDL2RemoteAllocator::HDDL2RemoteAllocator(HddlUnite::Device::Ptr& device): _devicePtr(device) {}

void* HDDL2RemoteAllocator::alloc(size_t size) noexcept {
    std::lock_guard<std::mutex> lock(memStorMutex);

    if (!isValidInput(size)) {
        return nullptr;
    }

    if (_devicePtr == nullptr) {
        printf("%s: Device ptr is null!\n", __FUNCTION__);
        return nullptr;
    }
    printf("%s: Allocate memory of %d size\n", __FUNCTION__, static_cast<int>(size));

    try {
        HddlUnite::SMM::RemoteMemory::Ptr remoteMemoryPtr = HddlUnite::SMM::allocate(*_devicePtr, size);
        HDDL2RemoteMemoryContainer memoryContainer;
        memoryContainer.remoteMemory = remoteMemoryPtr;
        _memoryStorage.emplace(static_cast<void*>(remoteMemoryPtr.get()), memoryContainer);

        return static_cast<void*>(remoteMemoryPtr.get());
    } catch (...) {
        return nullptr;
    }
}

bool HDDL2RemoteAllocator::free(void* handle) noexcept {
    std::lock_guard<std::mutex> lock(memStorMutex);

    if (handle == nullptr) {
        printf("%s: Invalid address: %p \n", __FUNCTION__, handle);
        return false;
    }
    auto iterator = _memoryStorage.find(handle);
    if (iterator == _memoryStorage.end()) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, handle);
        return false;
    }

    auto memory = &iterator->second;
    if (memory->isLocked) {
        printf("%s: Memory %p is locked!\n", __FUNCTION__, handle);
        return false;
    }

    printf("%s: Memory %p found, removing element\n", __FUNCTION__, handle);
    _memoryStorage.erase(iterator);
    return true;
}

void HDDL2RemoteAllocator::Release() noexcept { delete this; }

void* HDDL2RemoteAllocator::lock(void* handle, InferenceEngine::LockOp lockOp) noexcept {
    std::lock_guard<std::mutex> lock(memStorMutex);

    auto iterator = _memoryStorage.find(handle);
    if (iterator == _memoryStorage.end()) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, handle);
        return nullptr;
    }

    printf("%s: Locking memory %p \n", __FUNCTION__, handle);

    auto memory = &iterator->second;

    if (memory->isLocked) {
        printf("%s: Memory %p is already locked!\n", __FUNCTION__, handle);
        return nullptr;
    }

    memory->isLocked = true;
    memory->lockOp = lockOp;

    const size_t dmaBufSize = memory->remoteMemory->getBufSize();
    memory->localMemory.resize(dmaBufSize);

    if (dmaBufSize != memory->localMemory.size()) {
        printf("%s: dmaBufSize(%d) != memory->size(%d)\n", __FUNCTION__, static_cast<int>(dmaBufSize),
            static_cast<int>(memory->localMemory.size()));
        return nullptr;
    }
    printf("%s: Sync %d memory from device, handle %p\n", __FUNCTION__, static_cast<int>(memory->localMemory.size()),
        handle);

    if (memory->lockOp == InferenceEngine::LOCK_FOR_READ) {
        HddlStatusCode statusCode =
            memory->remoteMemory->syncFromDevice(memory->localMemory.data(), memory->localMemory.size());
        if (statusCode != HDDL_OK) {
            memory->isLocked = false;
            return nullptr;
        }
    }
    return memory->localMemory.data();
}

void HDDL2RemoteAllocator::unlock(void* handle) noexcept {
    std::lock_guard<std::mutex> lock(memStorMutex);

    auto iterator = _memoryStorage.find(handle);
    if (iterator == _memoryStorage.end() || !iterator->second.isLocked) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, handle);
        return;
    }
    auto memory = &iterator->second;

    if (memory->lockOp == InferenceEngine::LOCK_FOR_WRITE) {
        // Sync memory to device
        printf("%s: Sync %d memory to device, handle %p\n", __FUNCTION__, static_cast<int>(memory->localMemory.size()),
            handle);

        memory->remoteMemory->syncToDevice(memory->localMemory.data(), memory->localMemory.size());
    } else {
        printf("%s: LOCK_FOR_READ, Memory %d will NOT be synced, handle %p\n", __FUNCTION__,
            static_cast<int>(memory->localMemory.size()), handle);
    }

    memory->isLocked = false;
}
