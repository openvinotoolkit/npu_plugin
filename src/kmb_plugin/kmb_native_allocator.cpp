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

#include "kmb_native_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <zconf.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace vpu::KmbPlugin;

void* KmbNativeAllocator::alloc(size_t size) noexcept {
    void* virtAddr = nullptr;
    int fileDesc = -1;
    virtAddr = static_cast<unsigned char*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fileDesc, 0));

    if (virtAddr == MAP_FAILED) return nullptr;

    auto getShiftBase2 = [](long num) -> size_t {
        size_t shiftCount = 0;
        while (num >>= 1) shiftCount++;
        return shiftCount;
    };
    // HACK:
    // instead of storing physical address. Let's pack virtual address into uint32_t physAddr.
    // mmap aligns a pointer on pagesize. We can avoid storing this information
    uint32_t physAddr = reinterpret_cast<unsigned long>(virtAddr) >> getShiftBase2(getpagesize());

    MemoryDescriptor memDesc = {
        size,      // size
        fileDesc,  // file descriptor
        physAddr   // physical address
    };
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
}

bool KmbNativeAllocator::free(void* handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
    }

    auto memoryDesc = memoryIt->second;

    auto out = munmap(handle, memoryDesc.size);
    if (out == -1) {
        return false;
    }

    _allocatedMemory.erase(handle);

    return true;
}
