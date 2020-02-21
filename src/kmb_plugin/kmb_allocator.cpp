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

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <iostream>
#include <string>
#ifdef ENABLE_VPUAL
#include <vpusmm.h>
#endif

#include <iostream>
#include <string>
#include <vector>

#include "kmb_allocator.h"
#include "kmb_native_allocator.h"
#include "kmb_udma_allocator.h"
#include "kmb_vpusmm_allocator.h"

using namespace vpu::KmbPlugin;

void* KmbAllocator::lock(void* handle, InferenceEngine::LockOp) noexcept {
    if (_allocatedMemory.find(handle) == _allocatedMemory.end()) return nullptr;

    return handle;
}

void KmbAllocator::unlock(void* handle) noexcept { UNUSED(handle); }

unsigned long KmbAllocator::getPhysicalAddress(void* handle) noexcept {
    auto memoryIt = _allocatedMemory.find(handle);
    if (memoryIt == _allocatedMemory.end()) {
        return 0;
    }

    auto memoryDesc = memoryIt->second;
    return memoryDesc.physAddr;
}

bool KmbAllocator::isValidPtr(void* ptr) noexcept { return ptr != nullptr; }

void KmbAllocator::setSliceIdx(int sliceIdx) {
    if (sliceIdx >= 0 && sliceIdx < VPUSMM_SLICE_COUNT) {
        _sliceIdx = sliceIdx;
    } else {
        std::string sliceIdxStr = std::to_string(sliceIdx);
        std::string sliceRangeStr = "[0:" + std::to_string(VPUSMM_SLICE_COUNT) + ")";
        std::string errorMsg =
            "VPUSMMAllocator::VPUSMMAllocator: slice index " + sliceIdxStr + " is out of range " + sliceRangeStr;
        throw std::runtime_error(errorMsg);
    }
}

std::shared_ptr<KmbAllocator>& vpu::KmbPlugin::getKmbAllocator(int sliceIdx) {
    static std::vector<std::shared_ptr<KmbAllocator>> allocatorList(VPUSMM_SLICE_COUNT, nullptr);
    if (sliceIdx < VPUSMM_SLICE_COUNT && allocatorList.at(sliceIdx) == nullptr) {
        const char* allocatorEnvPtr = std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE");
        std::string allocatorType;
        if (allocatorEnvPtr) {
            allocatorType = allocatorEnvPtr;
        }
        if (allocatorType == "UDMA") {
            allocatorList[sliceIdx] = std::make_shared<KmbUdmaAllocator>();
        } else if (allocatorType == "NATIVE") {
            allocatorList[sliceIdx] = std::make_shared<KmbNativeAllocator>();
        } else {
            allocatorList[sliceIdx] = std::make_shared<KmbVpusmmAllocator>();
            // TODO allocator constructor must take slice index as an argument
            allocatorList.at(sliceIdx)->setSliceIdx(sliceIdx);
        }
    }
    return allocatorList.at(sliceIdx);
}
