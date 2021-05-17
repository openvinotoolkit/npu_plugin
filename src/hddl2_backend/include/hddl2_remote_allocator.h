//
// Copyright 2019-2020 Intel Corporation.
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

#pragma once
// System
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
// IE
#include "ie_allocator.hpp"
#include "ie_remote_context.hpp"
// Plugin
#include "vpux.hpp"
#include "vpux_config.hpp"
// Low-level
#include <HddlUnite.h>
#include <RemoteMemory.h>

namespace vpux {
namespace hddl2 {

//------------------------------------------------------------------------------
struct HDDL2RemoteMemoryContainer {
    explicit HDDL2RemoteMemoryContainer(const HddlUnite::RemoteMemory::Ptr& remoteMemory);

    InferenceEngine::LockOp lockOp = InferenceEngine::LOCK_FOR_WRITE;
    bool isLocked = false;

    std::vector<uint8_t> localMemory;
    HddlUnite::RemoteMemory::Ptr remoteMemory = nullptr;
};

//------------------------------------------------------------------------------
/**
 * @brief Hide all allocation and synchronization logic for HDDL2 device behind this class
 */
class HDDL2RemoteAllocator : public vpux::Allocator {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteAllocator>;

    explicit HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr,
                                  const vpu::LogLevel logLevel = vpu::LogLevel::None);

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
    void* incrementRemoteMemoryCounter(void* remoteMemoryHandle, const HddlUnite::eRemoteMemoryFormat format) noexcept;

private:
    HddlUnite::WorkloadContext::Ptr _contextPtr = nullptr;

    std::map<void*, HDDL2RemoteMemoryContainer> _memoryStorage;
    std::mutex memStorageMutex;
    const vpu::Logger::Ptr _logger;
    std::map<void*, size_t> _memoryHandleCounter;
};

}  // namespace hddl2
}  // namespace vpux
