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

#pragma once
#include <HddlUnite.h>
#include <RemoteMemory.h>

#include <atomic>
#include <ie_allocator.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "ie_remote_context.hpp"

// Emulator limit 4MB
constexpr size_t MAX_ALLOC_SIZE = static_cast<size_t>(0x1u << 22u);

namespace vpu {
namespace HDDL2Plugin {

using RemoteMemoryFD = uint64_t;

//------------------------------------------------------------------------------
//      struct HDDL2RemoteMemoryContainer
//------------------------------------------------------------------------------
struct HDDL2RemoteMemoryContainer {
    explicit HDDL2RemoteMemoryContainer(const HddlUnite::SMM::RemoteMemory::Ptr& remoteMemory);

    InferenceEngine::LockOp lockOp;
    bool isLocked = false;

    std::vector<uint8_t> localMemory;
    HddlUnite::SMM::RemoteMemory::Ptr remoteMemory = nullptr;
};

//------------------------------------------------------------------------------
//      class HDDL2RemoteAllocator
//------------------------------------------------------------------------------
/**
 * @brief Hide all allocation and synchronization logic for HDDL2 device behind this class
 */
class HDDL2RemoteAllocator : public InferenceEngine::IAllocator {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteAllocator>;

    explicit HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr);

    ~HDDL2RemoteAllocator() override = default;

    /**
     * @brief Lock memory and synchronize local buffer with remote
     * @return Pointer to local memory buffer
     */
    void* lock(void* remoteMemoryHandle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override;

    /**
     * @brief Unlock and synchronize memory from host to device if locked for write
     */
    void unlock(void* remoteMemoryHandle) noexcept override;

    // TODO alloc function should provide somehow bufFd of created memory, otherwise it's not
    //  possible to use it in another application.
    /**
     * @brief Allocate remote memory on device
     * @return Handle to allocated memory
     */
    void* alloc(size_t size) noexcept override;

    /**
     * @brief Wrap already allocated on device memory
     * @return Handle to allocated memory
     */
    void* wrapRemoteMemory(const RemoteMemoryFD& remoteMemoryFd, const size_t& size) noexcept;

    /**
     * @brief Free local memory and remote if we are owner
     * @return True if successful otherwise false
     */
    bool free(void* remoteMemoryHandle) noexcept override;

    /**
     * @brief Free all memory
     */
    void Release() noexcept override;

private:
    HddlUnite::WorkloadContext::Ptr _contextPtr = nullptr;

    std::map<void*, HDDL2RemoteMemoryContainer> _memoryStorage;
    std::mutex memStorageMutex;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
