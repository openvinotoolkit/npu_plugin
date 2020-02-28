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

#include "vpusmm.h"

#include "allocators.hpp"

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
    return requiredSize;
}

VPUSMMAllocator::VPUSMMAllocator(int sliceIdx) : _sliceIdx(sliceIdx) {
    if (_sliceIdx < 0 || _sliceIdx >= VPUSMM_SLICE_COUNT) {
        std::string sliceIdxStr = std::to_string(_sliceIdx);
        std::string sliceRangeStr = "[0:" + std::to_string(VPUSMM_SLICE_COUNT) + ")";
        std::string errorMsg = "VPUSMMAllocator::VPUSMMAllocator: slice index " + sliceIdxStr +
            " is out of range " + sliceRangeStr;
        throw std::runtime_error(errorMsg);
    }
}

void* VPUSMMAllocator::allocate(size_t requestedSize) {
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
    int fileDesc = vpurm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_COHERENT, _sliceIdx);
    if (fileDesc < 0) {
        throw std::runtime_error("VPUSMMAllocator::allocate: vpurm_alloc_dmabuf failed");
    }

    unsigned long physAddr = vpurm_import_dmabuf(fileDesc, VPU_DEFAULT, _sliceIdx);
    if (physAddr == 0) {
        throw std::runtime_error("VPUSMMAllocator::allocate: vpurm_import_dmabuf failed");
    }

    void* virtAddr = mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, fileDesc, _sliceIdx);
    if (virtAddr == MAP_FAILED) {
        throw std::runtime_error("VPUSMMAllocator::allocate: mmap failed");
    }
    std::tuple<int, void*, size_t> memChunk(fileDesc, virtAddr, requiredBlobSize);
    _memChunks.push_back(memChunk);

    return virtAddr;
}

void* VPUSMMAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
    std::tuple<int, void*, size_t> chunk = _memChunks.at(chunkIndex);
    void* virtAddr = std::get<1>(chunk);
    return virtAddr;
}

VPUSMMAllocator::~VPUSMMAllocator() {
    for (const std::tuple<int, void*, size_t> & chunk : _memChunks) {
        int fileDesc = std::get<0>(chunk);
        void* virtAddr = std::get<1>(chunk);
        size_t allocatedSize = std::get<2>(chunk);
        vpurm_unimport_dmabuf(fileDesc, _sliceIdx);
        munmap(virtAddr, allocatedSize);
        close(fileDesc);
    }
}

void* NativeAllocator::allocate(size_t requestedSize) {
    uint8_t* allocatedChunk = new uint8_t [requestedSize];
    _memChunks.push_back(allocatedChunk);
    return allocatedChunk;
}

void* NativeAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
    return _memChunks.at(chunkIndex);
}

NativeAllocator::~NativeAllocator() {
    for (uint8_t* chunk : _memChunks) {
        delete [] chunk;
    }
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
