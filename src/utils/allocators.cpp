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

#include <unistd.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "allocators.hpp"
#include "../kmb_plugin/kmb_allocator.h"
#include "ie_macro.hpp"

#if defined(__arm__) || defined(__aarch64__)
#include <vpusmm/vpusmm.h>
#endif

namespace vpu {

namespace KmbPlugin {

namespace utils {

int VPUSMMAllocator::_pageSize = getpagesize();

static uint32_t calculateRequiredSize(uint32_t blobSize, int pageSize) {
    uint32_t blobSizeRem = blobSize % pageSize;
    uint32_t requiredSize = (blobSize / pageSize) * pageSize;
    if (blobSizeRem) {
        requiredSize += pageSize;
    }
    // workaround for runtime bug. allocate at least two pages of memory
    // [Track number: h#18011677038]
    if (requiredSize < pageSize * 2) {
        requiredSize = pageSize * 2;
    }
    return requiredSize;
}

void* VPUSMMAllocator::allocate(size_t requestedSize) {
#if defined(__arm__) || defined(__aarch64__)
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
    int fileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_COHERENT);
    if (fileDesc < 0) {
        throw std::runtime_error("VPUSMMAllocator::allocate: vpusmm_alloc_dmabuf failed");
    }

    unsigned long physAddr = vpusmm_import_dmabuf(fileDesc, VPU_DEFAULT);
    if (physAddr == 0) {
        throw std::runtime_error("VPUSMMAllocator::allocate: vpusmm_import_dmabuf failed");
    }

    void* virtAddr = mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, fileDesc, 0);
    if (virtAddr == MAP_FAILED) {
        throw std::runtime_error("VPUSMMAllocator::allocate: mmap failed");
    }
    std::tuple<int, void*, size_t> memChunk(fileDesc, virtAddr, requiredBlobSize);
    _memChunks.push_back(memChunk);

    return virtAddr;
#else
    UNUSED(requestedSize);
    return nullptr;
#endif
}

void* VPUSMMAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
#if defined(__arm__) || defined(__aarch64__)
    std::tuple<int, void*, size_t> chunk = _memChunks.at(chunkIndex);
    void* virtAddr = std::get<1>(chunk);
    return virtAddr;
#else
    UNUSED(chunkIndex);
    return nullptr;
#endif
}

VPUSMMAllocator::~VPUSMMAllocator() {
#if defined(__arm__) || defined(__aarch64__)
    for (const std::tuple<int, void*, size_t> & chunk : _memChunks) {
        int fileDesc = std::get<0>(chunk);
        void* virtAddr = std::get<1>(chunk);
        size_t allocatedSize = std::get<2>(chunk);
        vpusmm_unimport_dmabuf(fileDesc);
        munmap(virtAddr, allocatedSize);
        close(fileDesc);
    }
#endif
}

void* NativeAllocator::allocate(size_t requestedSize) {
#if defined(__arm__) || defined(__aarch64__)
    uint8_t* allocatedChunk = new uint8_t [requestedSize];
    _memChunks.push_back(allocatedChunk);
    return allocatedChunk;
#else
    UNUSED(requestedSize);
    return nullptr;
#endif
}

void* NativeAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
#if defined(__arm__) || defined(__aarch64__)
    return _memChunks.at(chunkIndex);
#else
    UNUSED(chunkIndex);
    return nullptr;
#endif
}

NativeAllocator::~NativeAllocator() {
#if defined(__arm__) || defined(__aarch64__)
    for (uint8_t* chunk : _memChunks) {
        delete [] chunk;
    }
#endif
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
