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

#include "kmb_vpusmm_allocator.h"

#if defined(__arm__) || defined(__aarch64__)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vpusmm/vpusmm.h>
#endif

using namespace vpu::KmbPlugin;

void* KmbVpusmmAllocator::alloc(size_t size) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    size_t pageSize = getpagesize();
    size_t realSize = size + (size % pageSize ? (pageSize - size % pageSize) : 0);
    // workaround for runtime bug. allocate at least two pages of memory
    // [Track number: h#18011677038]
    if (realSize < pageSize * 2) {
        realSize = pageSize * 2;
    }

    auto fd = vpusmm_alloc_dmabuf(realSize, VPUSMMType::VPUSMMTYPE_COHERENT);

    auto physAddr = vpusmm_import_dmabuf(fd, VPU_DEFAULT);

    void* virtAddr = mmap(nullptr, realSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (virtAddr == MAP_FAILED) return nullptr;

    MemoryDescriptor memDesc = {
        realSize,  // size
        fd,        // file descriptor
        physAddr   // physical address
    };
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
#else
    UNUSED(size);
    return nullptr;
#endif
}

bool KmbVpusmmAllocator::free(void* handle) noexcept {
#if defined(__arm__) || defined(__aarch64__)
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
#else
    UNUSED(handle);
    return false;
#endif
}

bool KmbVpusmmAllocator::isValidPtr(void* ptr) noexcept {
#if defined(__arm__) || defined(__aarch64__)
    return ptr != nullptr && vpusmm_ptr_to_vpu(ptr) != 0;
#else
    UNUSED(ptr);
    return false;
#endif
}
