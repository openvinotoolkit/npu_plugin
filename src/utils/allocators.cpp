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

#ifdef __unix__
#include <unistd.h>

#include <sys/mman.h>
#else
#define getpagesize() 4096
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <stdexcept>

#include "allocators.hpp"
#include "ie_macro.hpp"

#if defined(__arm__) || defined(__aarch64__)
#include <vpusmm/vpusmm.h>
#endif

namespace vpu {

namespace KmbPlugin {

namespace utils {

uint32_t VPUSMMAllocator::_pageSize = getpagesize();

static uint32_t calculateRequiredSize(uint32_t blobSize, uint32_t pageSize) {
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
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
#if defined(__arm__) || defined(__aarch64__)
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
    UNUSED(requiredBlobSize);
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

int VPUSMMAllocator::getFileDescByVirtAddr(void* virtAddr) {
    auto virtAddrPredicate = [virtAddr](const std::tuple<int, void*, size_t> & chunk) -> bool {
        return virtAddr == std::get<1>(chunk);
    };

    auto memChunksIter = std::find_if(_memChunks.begin(), _memChunks.end(), virtAddrPredicate);
    if (memChunksIter == _memChunks.end()) {
        throw std::runtime_error("getFileDescByVirtAddrHelper: failed to find virtual address");
    }
    return std::get<0>(*memChunksIter);
}

int VPUSMMAllocator::allocateDMA(size_t requestedSize) {
#if defined(__arm__) || defined(__aarch64__)
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
    int fileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_COHERENT);
    if (fileDesc < 0) {
        throw std::runtime_error("VPUSMMAllocator::allocate: vpusmm_alloc_dmabuf failed");
    }
    std::tuple<int, void*, size_t> memChunk(fileDesc, nullptr, requiredBlobSize);
    _memChunks.push_back(memChunk);
    return fileDesc;
#else
    UNUSED(requestedSize);
    return -1;
#endif
}

void* VPUSMMAllocator::importDMA(const int& fileDesc) {
#if defined(__arm__) || defined(__aarch64__)
    auto fileDescPredicate = [fileDesc](const std::tuple<int, void*, size_t> & chunk) -> bool {
        return fileDesc == std::get<0>(chunk);
    };

    auto memChunksIter = std::find_if(_memChunks.begin(), _memChunks.end(), fileDescPredicate);
    if (memChunksIter == _memChunks.end()) {
        throw std::runtime_error("VPUSMMAllocator::importDMA: failed to find descriptor");
    }
    unsigned long physAddr = vpusmm_import_dmabuf(fileDesc, VPU_DEFAULT);
    if (physAddr == 0) {
        throw std::runtime_error("VPUSMMAllocator::importDMA: vpusmm_import_dmabuf failed");
    }
    const auto& requiredBlobSize = std::get<2>(*memChunksIter);
    void* virtAddr = mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, fileDesc, 0);
    if (virtAddr == MAP_FAILED) {
        throw std::runtime_error("VPUSMMAllocator::importDMA: mmap failed");
    }
    auto& chunkVirtAddr = std::get<1>(*memChunksIter);
    chunkVirtAddr = virtAddr;
    return virtAddr;
#else
    UNUSED(fileDesc);
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
        if (virtAddr != nullptr) {
            munmap(virtAddr, allocatedSize);
        }
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

int NativeAllocator::getFileDescByVirtAddr(void*) {
    return -1;
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
