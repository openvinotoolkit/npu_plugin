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

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "kmb_udma_allocator.h"

#include <iostream>
#include <string>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace vpu::KmbPlugin;

void *KmbUdmaAllocator::alloc(size_t size) noexcept {
    std::ostringstream bufferNameStream;
    int bufferCount = _allocatedMemory.size();
    bufferNameStream << "udmabuf" << bufferCount;
    const std::string bufname = bufferNameStream.str();
    const std::string udmabufdevname = "/dev/" + bufname;
    const std::string udmabufsize = "/sys/class/udmabuf/" +  bufname + "/size";
    const std::string udmabufphysaddr = "/sys/class/udmabuf/" + bufname + "/phys_addr";
    const std::string udmabufclassname = "/sys/class/udmabuf/" + bufname + "/sync_mode";

    // Set the sync mode.
    const std::string SYNC_MODE_STR = "3";
    int devFileDesc = -1;
    if ((devFileDesc  = open(udmabufclassname.c_str(), O_WRONLY | O_EXCL)) != -1) {
        std::size_t bytesWritten = write(devFileDesc, SYNC_MODE_STR.c_str(), SYNC_MODE_STR.size());
        UNUSED(bytesWritten);
        close(devFileDesc);
    } else {
        return nullptr;
    }

    std::size_t regionSize = size;
    // Get the size of the region.
    int bufSizeFileDesc = -1;
    if ((bufSizeFileDesc  = open(udmabufsize.c_str(), O_RDONLY | O_EXCL)) != -1) {
        const std::size_t maxRegionSizeLength = 1024;
        std::string regionSizeString(maxRegionSizeLength, 0x0);

        std::size_t bytesRead = read(bufSizeFileDesc, &regionSizeString[0], maxRegionSizeLength);
        UNUSED(bytesRead);
        std::istringstream regionStringToInt(regionSizeString);
        regionStringToInt >> regionSize;
        close(bufSizeFileDesc);
    } else {
        return nullptr;
    }

    // Get the physical address of the region.
    unsigned long physAddress = 0;
    int physAddrFileDesc = -1;
    if ((physAddrFileDesc  = open(udmabufphysaddr.c_str(), O_RDONLY | O_EXCL)) != -1) {
        const std::size_t maxPhysAddrLength = 1024;
        std::string physAddrString(maxPhysAddrLength, 0x0);

        std::size_t bytesRead = read(physAddrFileDesc, &physAddrString[0], maxPhysAddrLength);
        UNUSED(bytesRead);
        std::istringstream physAddrToHex(physAddrString);
        physAddrToHex >> std::hex >> physAddress;
        close(physAddrFileDesc);
    } else {
        return nullptr;
    }

    // Map a virtual address which we can use to the region.
    // O_SYNC is important to ensure our data is written through the cache.
    int fileDesc = -1;
    void *virtAddr = nullptr;
    if ((fileDesc = open(udmabufdevname.c_str(), O_RDWR | O_SYNC | O_EXCL)) != -1) {
        virtAddr = static_cast<unsigned char*>(mmap(nullptr, regionSize, PROT_READ|PROT_WRITE, MAP_SHARED, fileDesc, 0));
    } else {
        return nullptr;
    }

    if (virtAddr == MAP_FAILED) {
        close(fileDesc);
        return nullptr;
    }

    MemoryDescriptor memDesc = {
            regionSize,  // size
            fileDesc,    // file descriptor
            physAddress  // physical address
    };
    close(fileDesc);
    _allocatedMemory[virtAddr] = memDesc;

    return virtAddr;
}

bool KmbUdmaAllocator::free(void *handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return false;
    }

    auto memoryDesc = memoryIt->second;

    auto out = munmap(handle, memoryDesc.size);
    if (out == -1) {
        return false;
    }
    close(memoryDesc.fd);

    _allocatedMemory.erase(handle);

    return true;
}
