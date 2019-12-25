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

//------------------------------------------------------------------------------
//      struct HDDL2RemoteMemoryContainer
//------------------------------------------------------------------------------
struct HDDL2RemoteMemoryContainer {
    /**
     * @brief Type of lock operation
     */
    InferenceEngine::LockOp lockOp;

    bool isLocked = false;

    /**
     * @brief Smart pointer to remote memory
     */
    HddlUnite::SMM::RemoteMemory::Ptr remoteMemory = nullptr;

    /**
     * @brief Pointer to local memory
     */
    std::vector<uint8_t> localMemory;
};

//------------------------------------------------------------------------------
//      class HDDL2RemoteAllocator
//------------------------------------------------------------------------------
/**
 * @brief Hide all allocation and synchronization logic for HDDL2 device behind this class
 */
class HDDL2RemoteAllocator : public InferenceEngine::IAllocator {
public:
    /**
     * @brief Smart pointer to allocator
     */
    using Ptr = std::shared_ptr<HDDL2RemoteAllocator>;

    /**
     * @brief Create allocator based on device
     * @param device Smart pointer to HDDL Unite device
     */
    explicit HDDL2RemoteAllocator(HddlUnite::Device::Ptr& device);

    // TODO Destroy all associated RemoteBlobs on destruction?
    ~HDDL2RemoteAllocator() override = default;

    /**
     * @brief Lock memory and synchronize local buffer with remote
     * @param handle Remote memory handle
     * @return Pointer to local memory buffer
     */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override;

    /**
     * @brief Unlock and synchronize memory from host to device if locked for write
     * @param handle Remote memory handle
     */
    void unlock(void* handle) noexcept override;

    /**
     * @brief Allocate remote memory on device
     * @param size Memory to allocate
     * @return Handler to allocated memory
     */
    void* alloc(size_t size) noexcept override;

    /**
     * @brief Free up remote memory on device
     * @param Handler to allocated memory
     * @return True if successful otherwise false
     */
    bool free(void* handle) noexcept override;

    /**
     * @brief Free all memory
     */
    void Release() noexcept override;

private:
    HddlUnite::Device::Ptr _devicePtr;
    std::map<void*, HDDL2RemoteMemoryContainer> _memoryStorage;
    std::mutex memStorMutex;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
