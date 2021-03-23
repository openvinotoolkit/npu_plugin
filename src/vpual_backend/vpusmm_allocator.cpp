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

#include "vpusmm_allocator.hpp"

#include "vpux/utils/core/helper_macros.hpp"

#include <iostream>
#include <memory>

#include "vpux_params_private_options.h"

#if defined(__arm__) || defined(__aarch64__)
#include <sys/mman.h>
#include <unistd.h>
#include <vpumgr.h>
#endif

namespace vpux {

#if defined(__arm__) || defined(__aarch64__)
static size_t alignMemorySize(const size_t size) {
    size_t pageSize = getpagesize();
    size_t realSize = size + (size % pageSize ? (pageSize - size % pageSize) : 0);
    // workaround for runtime bug. allocate at least two pages of memory
    // [Track number: h#18011677038]
    if (realSize < pageSize * 2) {
        realSize = pageSize * 2;
    }
    return realSize;
}
#endif

// TODO Cut copy of KmbBlobParams (wo config, _logger). Refactor to avoid duplication.
class VPUSMMAllocatorParams {
public:
    explicit VPUSMMAllocatorParams(const InferenceEngine::ParamMap& paramMap);

    InferenceEngine::ParamMap getParamMap() const { return _paramMap; }
    KmbRemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; }
    KmbHandleParam getRemoteMemoryHandle() const { return _remoteMemoryHandle; }
    KmbOffsetParam getRemoteMemoryOffset() const { return _remoteMemoryOffset; }
    size_t getSize() const { return _size; }

protected:
    InferenceEngine::ParamMap _paramMap;
    KmbRemoteMemoryFD _remoteMemoryFd;
    KmbHandleParam _remoteMemoryHandle;
    KmbOffsetParam _remoteMemoryOffset;
    size_t _size;
};

VPUSMMAllocatorParams::VPUSMMAllocatorParams(const InferenceEngine::ParamMap& params): _paramMap(params) {
    if (params.empty()) {
        IE_THROW() << "VPUSMMAllocatorParams: Param map for blob is empty.";
    }

    auto sizeIter = params.find(InferenceEngine::KMB_PARAM_KEY(ALLOCATION_SIZE));
    if (sizeIter == params.end()) {
        IE_THROW() << "VPUSMMAllocatorParams: Size of allocation is not provided.";
    }
    try {
        _size = params.at(InferenceEngine::KMB_PARAM_KEY(ALLOCATION_SIZE)).as<size_t>();
    } catch (...) {
        IE_THROW() << "VPUSMMAllocatorParams: Failed to get size of allocation.";
    }

    auto remoteMemoryFdIter = params.find(InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD));
    if (remoteMemoryFdIter == params.end()) {
        IE_THROW() << "VPUSMMAllocatorParams: "
                           << "Param map does not contain remote memory file descriptor "
                              "information";
    }
    try {
        _remoteMemoryFd = remoteMemoryFdIter->second.as<KmbRemoteMemoryFD>();
    } catch (...) {
        IE_THROW() << "VPUSMMAllocatorParams: Remote memory fd param has incorrect type";
    }

    auto remoteMemoryHandleIter = params.find(InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE));
    auto remoteMemoryOffsetIter = params.find(InferenceEngine::KMB_PARAM_KEY(MEM_OFFSET));

    // memory handle is preferable
    if (remoteMemoryHandleIter != params.end()) {
        try {
            _remoteMemoryHandle = remoteMemoryHandleIter->second.as<KmbHandleParam>();
            _remoteMemoryOffset = 0;
        } catch (...) {
            IE_THROW() << "KmbBlobParams::KmbBlobParams: Remote memory handle param has incorrect type";
        }
    } else if (remoteMemoryOffsetIter != params.end()) {
        try {
            _remoteMemoryHandle = nullptr;
            _remoteMemoryOffset = remoteMemoryOffsetIter->second.as<KmbOffsetParam>();
        } catch (...) {
            IE_THROW() << "KmbBlobParams::KmbBlobParams: Remote memory offset param has incorrect type";
        }
    } else {
        IE_THROW() << "KmbBlobParams::KmbBlobParams: "
                           << "Param map should contain either remote memory handle "
                           << "or remote memory offset.";
    }
}

//------------------------------------------------------------------------------
VpusmmAllocator::VpusmmAllocator(const int deviceId): _deviceId(deviceId) {}

void* VpusmmAllocator::lock(void* handle, InferenceEngine::LockOp) noexcept {
    // isValidPtr check required when handle is allocated by external app via vpurm
    if (_allocatedMemory.find(handle) == _allocatedMemory.end() && !isValidPtr(handle)) return nullptr;

    return handle;
}

void VpusmmAllocator::unlock(void* handle) noexcept { VPUX_UNUSED(handle); }  // cpplint mark this line as false positive

unsigned long VpusmmAllocator::getPhysicalAddress(void* handle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    return vpurm_ptr_to_vpu(handle, _deviceId);
#else
    VPUX_UNUSED(handle);
    return 0;
#endif
}

bool VpusmmAllocator::isValidPtr(void* ptr) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    return ptr != nullptr && vpurm_ptr_to_vpu(ptr, _deviceId) != 0;
#else
    VPUX_UNUSED(ptr);
    return false;
#endif
}

void* VpusmmAllocator::wrapRemoteMemoryHandle(
    const KmbRemoteMemoryFD& remoteMemoryFd, const size_t size, void* memHandle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    auto physAddr = vpurm_ptr_to_vpu(memHandle, _deviceId);
    if (physAddr == 0) {
        physAddr = vpurm_import_dmabuf(remoteMemoryFd, VPU_DEFAULT, _deviceId);
    }

    MemoryDescriptor memDesc = {
        size,            // size
        remoteMemoryFd,  // file descriptor
        physAddr,        // physical address
        false            // memory wasn't allocated, it was imported
    };
    _allocatedMemory[memHandle] = memDesc;

    return memHandle;
#else
    VPUX_UNUSED(remoteMemoryFd);
    VPUX_UNUSED(size);
    VPUX_UNUSED(memHandle);
    return nullptr;
#endif
}

void* VpusmmAllocator::wrapRemoteMemoryOffset(
    const KmbRemoteMemoryFD& remoteMemoryFd, const size_t size, const KmbOffsetParam& memOffset) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    auto physAddr = vpurm_import_dmabuf(remoteMemoryFd, VPU_DEFAULT, _deviceId);
    size_t realSize = alignMemorySize(size + memOffset);
    // mmap always maps to the base physical address no matter which offset was provided
    // TODO find out whether that's expected
    void* virtAddr = mmap(nullptr, realSize, PROT_READ | PROT_WRITE, MAP_SHARED, remoteMemoryFd, 0);
    if (virtAddr == MAP_FAILED) return nullptr;

    // add offset and translate virtual address to physical
    virtAddr = reinterpret_cast<uint8_t*>(virtAddr) + memOffset;
    physAddr = vpurm_ptr_to_vpu(virtAddr, _deviceId);

    MemoryDescriptor memDesc = {
        size,            // size
        remoteMemoryFd,  // file descriptor
        physAddr,        // physical address
        false            // memory wasn't allocated, it was imported
    };
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
#else
    VPUX_UNUSED(remoteMemoryFd);
    VPUX_UNUSED(size);
    VPUX_UNUSED(memOffset);
    return nullptr;
#endif
}

void* VpusmmAllocator::alloc(size_t size) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    size_t realSize = alignMemorySize(size);

    auto fd = vpurm_alloc_dmabuf(realSize, VPUSMMType::VPUSMMTYPE_COHERENT, _deviceId);

    auto physAddr = vpurm_import_dmabuf(fd, VPU_DEFAULT, _deviceId);

    void* virtAddr = mmap(nullptr, realSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (virtAddr == MAP_FAILED) return nullptr;

    MemoryDescriptor memDesc = {
        realSize,  // size
        fd,        // file descriptor
        physAddr,  // physical address
        true       // memory was allocated
    };
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
#else
    VPUX_UNUSED(size);
    return nullptr;
#endif
}

bool VpusmmAllocator::free(void* handle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
    }
    if (!memoryIt->second.isMemoryOwner) {
        return true;
    }

    auto memoryDesc = memoryIt->second;

    vpurm_unimport_dmabuf(memoryDesc.fd, _deviceId);

    auto out = munmap(handle, memoryDesc.size);
    if (out == -1) {
        return false;
    }
    close(memoryDesc.fd);

    _allocatedMemory.erase(handle);

    return true;
#else
    VPUX_UNUSED(handle);
    return false;
#endif
}

VpusmmAllocator::~VpusmmAllocator() {
    if (!_allocatedMemory.empty()) {
        std::size_t allocatedChunksCount = 0;
        std::size_t amount = 0;
        for (const auto& p : _allocatedMemory) {
            // count only allocated chunks, skip imported chunks
            if (p.second.isMemoryOwner) {
                allocatedChunksCount++;
                amount += p.second.size;
            }
        }
        if (allocatedChunksCount > 0) {
            std::cerr << "Error: " << allocatedChunksCount << " memory chunks ";
            std::cerr << amount << " bytes amount were not freed!" << std::endl;
        }
    }
}

void* VpusmmAllocator::wrapRemoteMemory(const InferenceEngine::ParamMap& map) noexcept {
    std::lock_guard<std::mutex> lock(wrapMemoryMutex);
    VPUSMMAllocatorParams params(map);
    const auto& remoteMemoryFd = params.getRemoteMemoryFD();
    const auto& size = params.getSize();
    if (params.getRemoteMemoryHandle() != nullptr) {
        return wrapRemoteMemoryHandle(remoteMemoryFd, size, params.getRemoteMemoryHandle());
    } else {
        // fallback to offsets when memory handle is not specified
        return wrapRemoteMemoryOffset(remoteMemoryFd, size, params.getRemoteMemoryOffset());
    }
}

}  // namespace vpux
