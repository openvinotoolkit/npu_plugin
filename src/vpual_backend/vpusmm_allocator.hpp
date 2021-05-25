//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <ie_allocator.hpp>
#include <mutex>
#include <unordered_map>
#include <vpux.hpp>

#include "vpux/vpux_plugin_params.hpp"

namespace vpux {

class VpusmmAllocator : public Allocator {
public:
    using Ptr = std::shared_ptr<VpusmmAllocator>;
    VpusmmAllocator(const int deviceId);
    VpusmmAllocator(const VpusmmAllocator&) = delete;
    void* lock(void* handle, InferenceEngine::LockOp) noexcept override;

    void unlock(void* handle) noexcept override;

    virtual void* alloc(const size_t size) noexcept override;

    virtual bool free(void* handle) noexcept override;

    unsigned long getPhysicalAddress(void* handle) noexcept override;

    virtual bool isValidPtr(void* ptr) noexcept;

    void* wrapRemoteMemory(const InferenceEngine::ParamMap& map) noexcept override;
    // TODO Deprecated, remove when will be possible
    void* wrapRemoteMemoryHandle(
        const VpuxRemoteMemoryFD& remoteMemoryFd, const size_t size, void* memHandle) noexcept override;
    void* wrapRemoteMemoryOffset(
        const VpuxRemoteMemoryFD& remoteMemoryFd, const size_t size, const VpuxOffsetParam& memOffset) noexcept override;
    virtual ~VpusmmAllocator();

protected:
    std::mutex wrapMemoryMutex;

    struct MemoryDescriptor {
        size_t size;
        int fd;
        unsigned long physAddr;
        bool isMemoryOwner;
    };
    std::unordered_map<void*, MemoryDescriptor> _allocatedMemory;
    int _deviceId = 0;  // signed integer to be consistent with vpurm API
};

}  // namespace vpux
