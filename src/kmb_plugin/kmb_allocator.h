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

#ifndef UNUSED
#define UNUSED(var) (void)var
#endif

#include <ie_allocator.hpp>
#include <unordered_map>

namespace vpu {
namespace KmbPlugin {

class KmbAllocator : public InferenceEngine::IAllocator {
public:
    void* lock(void* handle, InferenceEngine::LockOp) noexcept override;

    void unlock(void* handle) noexcept override;

    virtual void* alloc(size_t size) noexcept = 0;

    virtual bool free(void* handle) noexcept = 0;

    void Release() noexcept override {}

    unsigned long getPhysicalAddress(void* handle) noexcept;

    virtual bool isValidPtr(void* ptr) noexcept;

    virtual ~KmbAllocator() = default;

protected:
    struct MemoryDescriptor {
        size_t size;
        int fd;
        unsigned long physAddr;
    };
    std::unordered_map<void*, MemoryDescriptor> _allocatedMemory;
};

std::shared_ptr<KmbAllocator>& getKmbAllocator();

}  // namespace KmbPlugin
}  // namespace vpu
