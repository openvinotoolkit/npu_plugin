//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

// System
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

// IE
#include <ie_allocator.hpp>
#include <ie_remote_context.hpp>

// Plugin
#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

// Low-level
#include <HddlUnite.h>
#include <RemoteMemory.h>

namespace vpux {
namespace hddl2 {

//------------------------------------------------------------------------------
struct HDDL2RemoteMemoryContainer {
    explicit HDDL2RemoteMemoryContainer(const HddlUnite::RemoteMemory::Ptr& remoteMemory);

    InferenceEngine::LockOp _lockOp = InferenceEngine::LOCK_FOR_WRITE;
    bool _isLocked = false;

    std::vector<uint8_t> _localMemory;
    HddlUnite::RemoteMemory::Ptr _remoteMemory = nullptr;
    void* _updatedMemoryHandle = nullptr;
};

//------------------------------------------------------------------------------
/**
 * @brief Hide all allocation and synchronization logic for HDDL2 device behind this class
 */
class HDDL2RemoteAllocator : public vpux::Allocator {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteAllocator>;

    explicit HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr,
                                  const LogLevel logLevel = LogLevel::None);

    /**
     * @brief Lock memory and synchronize local buffer with remote
     * @return Pointer to local memory buffer
     */
    void* lock(void* remoteMemoryHandle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override;

    /** @brief Unlock and synchronize memory from host to device if locked for write */
    void unlock(void* remoteMemoryHandle) noexcept override;

    /**
     * @brief Allocate remote memory on device (not implemented)
     * @return Handle to allocated memory
     */
    void* alloc(size_t size) noexcept override;

    /**
     * @brief Free local memory and remote if we are owner
     * @return True if successful otherwise false
     */
    bool free(void* remoteMemoryHandle) noexcept override;

    /**
     * @brief Wrap already allocated on device memory
     * @return Allocated remote memory
     */
    void* wrapRemoteMemory(const InferenceEngine::ParamMap& map) noexcept override;

    // TODO To remove. Specific KMB API
    void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size, void* memHandle) noexcept override;

    void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                 const size_t& memOffset) noexcept override;

    unsigned long getPhysicalAddress(void* handle) noexcept override;

protected:
    /**
     * @brief Fake free of already allocated on device memory by decrementing remote memory counter
     * @return Number of references on remote memory
     */
    size_t decrementRemoteMemoryCounter(void* remoteMemoryHandle, bool& findMemoryHandle) noexcept;

    /**
     * @brief Fake copy of already allocated on device memory by incrementing remote memory counter
     * @return Handle to allocated memory
     */
    void* incrementRemoteMemoryCounter(void* remoteMemoryHandle) noexcept;

    /**
     * @brief Free local memory and remote if we are owner
     * @return True if successful otherwise false
     */
    bool freeMemory(void* remoteMemoryHandle) noexcept;

private:
    HddlUnite::WorkloadContext::Ptr _contextPtr = nullptr;

    std::unordered_map<void*, HDDL2RemoteMemoryContainer> _memoryStorage;
    std::mutex memStorageMutex;
    Logger _logger;
    std::unordered_map<void*, size_t> _memoryHandleCounter;
    std::unordered_map<void*, void*> _updatedMemoryHandle;
};

}  // namespace hddl2
}  // namespace vpux
