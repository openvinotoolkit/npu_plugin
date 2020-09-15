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

#include <climits>
#include <memory>
#include <string>

using namespace vpu::HDDL2Plugin;
constexpr size_t MAX_ALLOC_SIZE = 1024 * 1024 * 1024;  // 1GB

bool static isValidRemoteMemory(const HddlUnite::RemoteMemory::Ptr& remoteMemory) {
    return remoteMemory->getDmaBufFd() != INT32_MAX;
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

HDDL2RemoteMemoryContainer::HDDL2RemoteMemoryContainer(const HddlUnite::RemoteMemory::Ptr& remoteMemory)
    : remoteMemory(remoteMemory) {}

HDDL2RemoteAllocator::HDDL2RemoteAllocator(
    const HddlUnite::WorkloadContext::Ptr& contextPtr, const vpu::HDDL2Config& config)
    : _config(config), _logger(std::make_shared<Logger>("RemoteAllocator", config.logLevel(), consoleOutput())) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Context pointer is null";
    }

    _contextPtr = contextPtr;
}

void* HDDL2RemoteAllocator::alloc(size_t size) noexcept {
    UNUSED(size);
    _logger->error("%s: not implemented!\n", __FUNCTION__);
    return nullptr;
}

void* HDDL2RemoteAllocator::wrapRemoteMemory(const HddlUnite::RemoteMemory::Ptr& remoteMemory) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (!remoteMemory) {
        return nullptr;
    }

    if (!isValidRemoteMemory(remoteMemory)) {
        _logger->warning("%s: Incorrect memory fd!\n", __FUNCTION__);
        return nullptr;
    }

    try {
        // Use already allocated memory
        HDDL2RemoteMemoryContainer memoryContainer(remoteMemory);
        void* remMemHandle = static_cast<void*>(remoteMemory.get());
        _memoryStorage.emplace(remMemHandle, memoryContainer);
        ++_memoryHandleCounter[remMemHandle];

        _logger->info("%s: Wrapped memory of %lu size\n", __FUNCTION__, remoteMemory->getMemoryDesc().getDataSize());
        return static_cast<void*>(remoteMemory.get());
    } catch (const std::exception& ex) {
        _logger->error("%s: Failed to wrap memory. Error: %s\n", __FUNCTION__, ex.what());
        return nullptr;
    }
}

void* HDDL2RemoteAllocator::incrementRemoteMemoryCounter(const void* remoteMemoryHandle) noexcept {
    if (remoteMemoryHandle == nullptr) {
        _logger->warning("%s: Invalid address: %p \n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(memStorageMutex);
    auto counter_it = _memoryHandleCounter.find(const_cast<void*>(remoteMemoryHandle));
    if (counter_it == _memoryHandleCounter.end()) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    ++_memoryHandleCounter[const_cast<void*>(remoteMemoryHandle)];
    return const_cast<void*>(remoteMemoryHandle);
}

size_t HDDL2RemoteAllocator::decrementRemoteMemoryCounter(void* remoteMemoryHandle, bool& findMemoryHandle) noexcept {
    auto counter_it = _memoryHandleCounter.find(remoteMemoryHandle);
    if (counter_it == _memoryHandleCounter.end()) {
        findMemoryHandle = false;
        return 0;
    }

    if (!counter_it->second) {
        findMemoryHandle = false;
        return 0;
    }

    findMemoryHandle = true;
    auto ret_counter = --(counter_it->second);
    if (!ret_counter) {
        _memoryHandleCounter.erase(counter_it);
    }
    return ret_counter;
}

bool HDDL2RemoteAllocator::free(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (remoteMemoryHandle == nullptr) {
        _logger->warning("%s: Invalid address: %p \n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }
    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    auto memory = &iterator->second;
    if (memory->isLocked) {
        _logger->warning("%s: Memory %p is locked!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    bool findMemoryHandle;
    auto handle_counter = decrementRemoteMemoryCounter(remoteMemoryHandle, findMemoryHandle);
    if (!findMemoryHandle) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    if (handle_counter) {
        _logger->info(
            "%s: Memory %p found, remaining references = %lu\n", __FUNCTION__, remoteMemoryHandle, handle_counter);
        return true;
    }

    _logger->info("%s: Memory %p found, removing element\n", __FUNCTION__, remoteMemoryHandle);
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
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    _logger->info("%s: Locking memory %p \n", __FUNCTION__, remoteMemoryHandle);

    auto memory = &iterator->second;

    if (memory->isLocked) {
        _logger->warning("%s: Memory %p is already locked!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    memory->isLocked = true;
    memory->lockOp = lockOp;

    const size_t dmaBufSize = memory->remoteMemory->getMemoryDesc().getDataSize();
    memory->localMemory.resize(dmaBufSize);

    if (dmaBufSize != memory->localMemory.size()) {
        _logger->info("%s: dmaBufSize(%d) != memory->size(%d)\n", __FUNCTION__, static_cast<int>(dmaBufSize),
            static_cast<int>(memory->localMemory.size()));
        return nullptr;
    }

    _logger->info("%s: LockOp: %s\n", __FUNCTION__, lockOpToStr(lockOp).c_str());

    // TODO Do this step only on R+W and R operations, not for Write
    _logger->info("%s: Sync %d memory from device, remoteMemoryHandle %p, fd %d\n", __FUNCTION__,
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
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return;
    }
    auto memory = &iterator->second;

    if (memory->lockOp == InferenceEngine::LOCK_FOR_WRITE) {
        // Sync memory to device
        _logger->info("%s: Sync %d memory to device, remoteMemoryHandle %p\n", __FUNCTION__,
            static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
        memory->remoteMemory->syncToDevice(memory->localMemory.data(), memory->localMemory.size());
    } else {
        _logger->warning("%s: LOCK_FOR_READ, Memory %d will NOT be synced, remoteMemoryHandle %p\n", __FUNCTION__,
            static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
    }

    memory->isLocked = false;
}
