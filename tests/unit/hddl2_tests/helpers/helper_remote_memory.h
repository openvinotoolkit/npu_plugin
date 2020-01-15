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

#pragma once

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
//      class RemoteMemory_Helper
//------------------------------------------------------------------------------
using RemoteMemoryFd = int;

class RemoteMemory_Helper {
public:
    RemoteMemoryFd allocateRemoteMemory(const WorkloadID &id, const size_t& size);
    void destroyRemoteMemory();

    std::string getRemoteMemory(const size_t &size);

    RemoteMemoryFd getMemoryFd();

private:
    HddlUnite::SMM::RemoteMemory::Ptr _memory = nullptr;
    RemoteMemoryFd _memoryFd = 0;

};

//------------------------------------------------------------------------------
//      class RemoteMemory_Helper Implementation
//------------------------------------------------------------------------------
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

    printf("Memory fd: %d\n", _memoryFd);
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
    char tempBuffer[MAX_ALLOC_SIZE] = {};
    auto retCode = _memory->syncFromDevice(tempBuffer, size);
    if (retCode != HDDL_OK) {
        return "";
    }
    return std::string(tempBuffer);
}
}  // namespace HDDL2Plugin
}  // namespace vpu
