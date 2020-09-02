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

#include <iostream>
#include <memory>

#if defined(__arm__) || defined(__aarch64__)
#include <sys/mman.h>
#include <unistd.h>
#include <vpusmm/vpusmm.h>
#endif

#include "ie_macro.hpp"

namespace vpux {

#if defined(__arm__) || defined(__aarch64__)
static size_t alignMemorySize(const size_t& size) {
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

VpusmmAllocator::VpusmmAllocator(const int& deviceId): _deviceId(deviceId) {}

void* VpusmmAllocator::lock(void* handle, InferenceEngine::LockOp) noexcept {
    if (_allocatedMemory.find(handle) == _allocatedMemory.end()) return nullptr;

    return handle;
}

void VpusmmAllocator::unlock(void* handle) noexcept { UNUSED(handle); }  // cpplint mark this line as false positive

unsigned long VpusmmAllocator::getPhysicalAddress(void* handle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    return vpurm_ptr_to_vpu(handle, _deviceId);
#else
    UNUSED(handle);
    return 0;
#endif
}

bool VpusmmAllocator::isValidPtr(void* ptr) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    return ptr != nullptr && vpurm_ptr_to_vpu(ptr, _deviceId) != 0;
#else
    UNUSED(ptr);
    return false;
#endif
}

void* VpusmmAllocator::wrapRemoteMemoryHandle(
    const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, void* memHandle) noexcept {
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
    UNUSED(remoteMemoryFd);
    UNUSED(size);
    UNUSED(memHandle);
    return nullptr;
#endif
}

void* VpusmmAllocator::wrapRemoteMemoryOffset(
    const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, const KmbOffsetParam& memOffset) noexcept {
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
    UNUSED(remoteMemoryFd);
    UNUSED(size);
    UNUSED(memOffset);
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
    UNUSED(size);
    return nullptr;
#endif
}

bool VpusmmAllocator::free(void* handle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
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
    UNUSED(handle);
    return false;
#endif
}

VpusmmAllocator::~VpusmmAllocator() {
    if (!_allocatedMemory.empty()) {
        std::size_t allocatedChunksCount = 0;
        std::size_t amount = 0;
        for (const auto& p : _allocatedMemory) {
            // count only allocated chunks, skip imported chunks
            if (p.second.isAllocated) {
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

}  // namespace vpux
