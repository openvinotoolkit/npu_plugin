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

#include <vpusmm.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>


#include "kmb_allocator.h"

#include <iostream>

using namespace vpu::KmbPlugin;

void *KmbAllocator::lock(void *handle, InferenceEngine::LockOp) noexcept {
    if (_allocatedMemory.find(handle) == _allocatedMemory.end())
        return nullptr;

    return handle;
}

void KmbAllocator::unlock(void *handle) noexcept {
    UNUSED(handle);
}

void *KmbAllocator::alloc(size_t size) noexcept {
    long pageSize = getpagesize();
    size_t realSize = size + (size % pageSize ? (pageSize - size % pageSize) : 0);

    auto fd = vpusmm_alloc_dmabuf(realSize, VPUSMMType::VPUSMMTYPE_COHERENT);

    auto physAddr = vpusmm_import_dmabuf(fd, DMA_BIDIRECTIONAL);

    void *virtAddr = mmap(nullptr, realSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (virtAddr == MAP_FAILED)
        return nullptr;

    MemoryDescriptor memDesc = {
            realSize,  // size
            fd,        // file descriptor
            physAddr   // physical address
    };
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
}

bool KmbAllocator::free(void *handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
    }

    auto memoryDesc = memoryIt->second;

    vpusmm_unimport_dmabuf(memoryDesc.fd);

    auto out = munmap(handle, memoryDesc.size);
    if (out == -1) {
        return false;
    }
    close(memoryDesc.fd);

    _allocatedMemory.erase(handle);

    return true;
}

unsigned long KmbAllocator::getPhysicalAddress(void *handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return 0;
    }

    auto memoryDesc = memoryIt->second;
    return memoryDesc.physAddr;
}
