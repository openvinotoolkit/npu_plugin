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

#include "hddl2_remote_allocator.h"

#include <memory>
#include <string>

using namespace vpu::HDDL2Plugin;

bool static isValidAllocateSize(size_t size) noexcept {
    if (size <= 0 || size > MAX_ALLOC_SIZE) {
        printf("%s: Incorrect size!\n", __FUNCTION__);
        return false;
    }
    return true;
}

bool static isValidRemoteMemoryFD(const RemoteMemoryFD& remoteMemoryFd) {
    if (remoteMemoryFd == UINT64_MAX) {
        printf("%s: Incorrect memory fd!\n", __FUNCTION__);
        return false;
    }
    return true;
}

static std::string lockOpToStr(const InferenceEngine::LockOp& lockOp) {
    switch (lockOp) {
    case InferenceEngine::LOCK_FOR_READ:
        return "LOCK_FOR_READ";
    case InferenceEngine::LOCK_FOR_WRITE:
        return "LOCK_FOR_WRITE (Read&Write)";
    default:
        return "Unknown Op Mode";
    }
}

HDDL2RemoteMemoryContainer::HDDL2RemoteMemoryContainer(const HddlUnite::SMM::RemoteMemory::Ptr& remoteMemory)
    : remoteMemory(remoteMemory) {}

HDDL2RemoteAllocator::HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Context pointer is null";
    }
    _contextPtr = contextPtr;
}

void* HDDL2RemoteAllocator::alloc(size_t size) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (!isValidAllocateSize(size)) {
        return nullptr;
    }

    try {
        HddlUnite::SMM::RemoteMemory::Ptr remoteMemoryPtr = HddlUnite::SMM::allocate(*_contextPtr, size);
        if (remoteMemoryPtr == nullptr) {
            THROW_IE_EXCEPTION << "Failed to allocate memory";
        }

        HDDL2RemoteMemoryContainer memoryContainer(remoteMemoryPtr);
        _memoryStorage.emplace(static_cast<void*>(remoteMemoryPtr.get()), memoryContainer);

        printf("%s: Allocate memory of %d size\n", __FUNCTION__, static_cast<int>(size));
        return static_cast<void*>(remoteMemoryPtr.get());
    } catch (const std::exception& ex) {
        printf("%s: Failed to allocate memory. Error: %s\n", __FUNCTION__, ex.what());
        return nullptr;
    }
}

void* HDDL2RemoteAllocator::wrapRemoteMemory(const RemoteMemoryFD& remoteMemoryFd, const size_t& size) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (!isValidAllocateSize(size) || !isValidRemoteMemoryFD(remoteMemoryFd)) {
        return nullptr;
    }

    try {
        // Use already allocated memory
        HddlUnite::SMM::RemoteMemory::Ptr remoteMemoryPtr =
            std::make_shared<HddlUnite::SMM::RemoteMemory>(*_contextPtr, remoteMemoryFd, size);

        HDDL2RemoteMemoryContainer memoryContainer(remoteMemoryPtr);
        _memoryStorage.emplace(static_cast<void*>(remoteMemoryPtr.get()), memoryContainer);

        printf("%s: Wrapped memory of %d size\n", __FUNCTION__, static_cast<int>(size));
        return static_cast<void*>(remoteMemoryPtr.get());
    } catch (const std::exception& ex) {
        printf("%s: Failed to wrap memory. Error: %s\n", __FUNCTION__, ex.what());
        return nullptr;
    }
}

bool HDDL2RemoteAllocator::free(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (remoteMemoryHandle == nullptr) {
        printf("%s: Invalid address: %p \n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }
    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    auto memory = &iterator->second;
    if (memory->isLocked) {
        printf("%s: Memory %p is locked!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    printf("%s: Memory %p found, removing element\n", __FUNCTION__, remoteMemoryHandle);
    _memoryStorage.erase(iterator);
    return true;
}

void HDDL2RemoteAllocator::Release() noexcept { delete this; }

// TODO LOCK_FOR_READ behavior when we will have lock for read-write
/**
 * LOCK_FOR_READ - do not sync to device on this call
 * LOCK_FOR_WRITE - default behavior - read&write option
 */
void* HDDL2RemoteAllocator::lock(void* remoteMemoryHandle, InferenceEngine::LockOp lockOp) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    printf("%s: Locking memory %p \n", __FUNCTION__, remoteMemoryHandle);

    auto memory = &iterator->second;

    if (memory->isLocked) {
        printf("%s: Memory %p is already locked!\n", __FUNCTION__, remoteMemoryHandle);
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

    printf("%s: LockOp: %s\n", __FUNCTION__, lockOpToStr(lockOp).c_str());

    // TODO Do this step only on R+W and R operations, not for Write
    printf("%s: Sync %d memory from device, remoteMemoryHandle %p, fd %d\n", __FUNCTION__,
        static_cast<int>(memory->localMemory.size()), remoteMemoryHandle, memory->remoteMemory->getDmaBufFd());

    HddlStatusCode statusCode =
        memory->remoteMemory->syncFromDevice(memory->localMemory.data(), memory->localMemory.size());
    if (statusCode != HDDL_OK) {
        memory->isLocked = false;
        return nullptr;
    }

    return memory->localMemory.data();
}

void HDDL2RemoteAllocator::unlock(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end() || !iterator->second.isLocked) {
        printf("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return;
    }
    auto memory = &iterator->second;

    if (memory->lockOp == InferenceEngine::LOCK_FOR_WRITE) {
        // Sync memory to device
        printf("%s: Sync %d memory to device, remoteMemoryHandle %p\n", __FUNCTION__,
            static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
        memory->remoteMemory->syncToDevice(memory->localMemory.data(), memory->localMemory.size());
    } else {
        printf("%s: LOCK_FOR_READ, Memory %d will NOT be synced, remoteMemoryHandle %p\n", __FUNCTION__,
            static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
    }

    memory->isLocked = false;
}
