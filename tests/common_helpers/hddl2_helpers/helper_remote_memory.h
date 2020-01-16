//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include "WorkloadContext.h"
#include "RemoteMemory.h"
#include "ie_layouts.h"

//------------------------------------------------------------------------------
//      class RemoteMemory_Helper
//------------------------------------------------------------------------------
#include <ie_algorithm.hpp>

using RemoteMemoryFd = uint64_t ;
// Emulator limit 4MB
constexpr size_t EMULATOR_MAX_ALLOC_SIZE = static_cast<size_t>(0x1u << 22u);

class RemoteMemory_Helper {
public:
    RemoteMemoryFd allocateRemoteMemory(const WorkloadID &id, const size_t& size);
    RemoteMemoryFd allocateRemoteMemory(const WorkloadID &id,
            const InferenceEngine::TensorDesc& tensorDesc);
    void destroyRemoteMemory();

    std::string getRemoteMemory(const size_t &size);
    bool isRemoteTheSame(const std::string &dataToCompare);
    void setRemoteMemory(const std::string& dataToSet);

    RemoteMemoryFd getMemoryFd();

    virtual ~RemoteMemory_Helper();

private:
    HddlUnite::SMM::RemoteMemory::Ptr _memory = nullptr;
    RemoteMemoryFd _memoryFd = 0;

};

//------------------------------------------------------------------------------
//      class RemoteMemory_Helper Implementation
//------------------------------------------------------------------------------
inline RemoteMemory_Helper::~RemoteMemory_Helper() {
    destroyRemoteMemory();
}

inline RemoteMemoryFd RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID &id,
                                                                const InferenceEngine::TensorDesc& tensorDesc) {
    const size_t size = InferenceEngine::details::product(
            tensorDesc.getDims().begin(), tensorDesc.getDims().end());
    return allocateRemoteMemory(id, size);
}

inline RemoteMemoryFd
RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID &id, const size_t &size) {
    if (_memory != nullptr) {
        printf("Memory already allocated!\n");
        return 0;
    }

    HddlUnite::WorkloadContext::Ptr context = HddlUnite::queryWorkloadContext(id);
    if (context == nullptr) {
        printf("Incorrect workload id!\n");
        return 0;
    }

    _memory = HddlUnite::SMM::allocate(*context, size);
    _memoryFd = _memory->getDmaBufFd();

    printf("Memory fd: %lu\n", _memoryFd);
    return _memoryFd;
}

inline void RemoteMemory_Helper::destroyRemoteMemory() {
    _memoryFd = 0;
    _memory = nullptr;
}

inline RemoteMemoryFd RemoteMemory_Helper::getMemoryFd() {
    return _memoryFd;
}

inline std::string RemoteMemory_Helper::getRemoteMemory(const size_t &size) {
    char tempBuffer[EMULATOR_MAX_ALLOC_SIZE] = {};
    auto retCode = _memory->syncFromDevice(tempBuffer, size);
    if (retCode != HDDL_OK) {
        printf("[ERROR] Failed to sync memory from device!\n");
        return "";
    }
    return std::string(tempBuffer);
}

inline bool RemoteMemory_Helper::isRemoteTheSame(const std::string &dataToCompare) {
    const size_t size = dataToCompare.size();
    const std::string remoteMemory = getRemoteMemory(size);
    if (dataToCompare != remoteMemory) {
        std::cout << "Handle: " << _memoryFd << " Remote memory " << remoteMemory
                     << " != local memory " << dataToCompare << std::endl;
        return false;
    }
    return true;
}

inline void RemoteMemory_Helper::setRemoteMemory(const std::string& dataToSet) {
    _memory->syncToDevice(dataToSet.data(), dataToSet.size());
}
