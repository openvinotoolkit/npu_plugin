//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "allocators.hpp"

#include "vpux/utils/core/helper_macros.hpp"

#ifdef __unix__
#include <sys/mman.h>
#include <unistd.h>
#else
#define getpagesize() 4096
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <stdexcept>

namespace vpu {
namespace KmbPlugin {
namespace utils {

uint32_t VPUSMMAllocator::_pageSize = getpagesize();

VPUSMMAllocator::VPUSMMAllocator(const int deviceId): _deviceId(deviceId) {
}

void* VPUSMMAllocator::allocate(size_t requestedSize) {
    VPUX_UNUSED(_deviceId);  // workaround for unused private field warning
    VPUX_UNUSED(requestedSize);
    return nullptr;
}

void* VPUSMMAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
    VPUX_UNUSED(chunkIndex);
    return nullptr;
}

int VPUSMMAllocator::getFileDescByVirtAddr(void* virtAddr) {
    auto virtAddrPredicate = [virtAddr](const std::tuple<int, void*, size_t>& chunk) -> bool {
        return virtAddr == std::get<1>(chunk);
    };

    auto memChunksIter = std::find_if(_memChunks.begin(), _memChunks.end(), virtAddrPredicate);
    if (memChunksIter == _memChunks.end()) {
        throw std::runtime_error("getFileDescByVirtAddrHelper: failed to find virtual address");
    }
    return std::get<0>(*memChunksIter);
}

int VPUSMMAllocator::allocateDMA(size_t requestedSize) {
    VPUX_UNUSED(requestedSize);
    return -1;
}

void* VPUSMMAllocator::importDMA(const int& fileDesc) {
    VPUX_UNUSED(fileDesc);
    return nullptr;
}

bool VPUSMMAllocator::free(void* handle) {
    bool isFound = false;
    VPUX_UNUSED(handle);
    return isFound;
}

VPUSMMAllocator::~VPUSMMAllocator() {
}

void* NativeAllocator::allocate(size_t requestedSize) {
    VPUX_UNUSED(requestedSize);
    return nullptr;
}

void* NativeAllocator::getAllocatedChunkByIndex(size_t chunkIndex) {
    VPUX_UNUSED(chunkIndex);
    return nullptr;
}

int NativeAllocator::getFileDescByVirtAddr(void*) {
    return -1;
}

NativeAllocator::~NativeAllocator() {
}

}  // namespace utils
}  // namespace KmbPlugin
}  // namespace vpu
