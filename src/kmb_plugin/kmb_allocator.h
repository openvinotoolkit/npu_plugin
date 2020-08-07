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

#include <ie_allocator.hpp>
#include <unordered_map>

#include "vpu/kmb_params.hpp"

namespace vpu {
namespace KmbPlugin {

class KmbAllocator : public InferenceEngine::IAllocator {
public:
    using Ptr = std::shared_ptr<KmbAllocator>;
    KmbAllocator(const int& deviceId);
    void* lock(void* handle, InferenceEngine::LockOp) noexcept override;

    void unlock(void* handle) noexcept override;

    void* alloc(size_t size) noexcept override;

    bool free(void* handle) noexcept override;

    void Release() noexcept override {}

    virtual unsigned long getPhysicalAddress(void* handle) noexcept;

    virtual bool isValidPtr(void* ptr) noexcept;

    virtual ~KmbAllocator();

    virtual void* wrapRemoteMemoryHandle(
        const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, void* memHandle) noexcept;
    virtual void* wrapRemoteMemoryOffset(
        const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, const KmbOffsetParam& memOffset) noexcept;

protected:
    struct MemoryDescriptor {
        size_t size;
        int fd;
        unsigned long physAddr;
        bool isAllocated;
    };
    std::unordered_map<void*, MemoryDescriptor> _allocatedMemory;
    int _deviceId = 0;  // signed integer to be consistent with vpurm API
};

}  // namespace KmbPlugin
}  // namespace vpu
