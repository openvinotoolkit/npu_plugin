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
#include <ie_algorithm.hpp>

class RemoteMemory_Helper {
public:
    using Ptr = std::shared_ptr<RemoteMemory_Helper>;

    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(const WorkloadID &id, const size_t& size);
    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(const WorkloadID &id,
            const InferenceEngine::TensorDesc& tensorDesc);
    void destroyRemoteMemory();

    std::string getRemoteMemory(const size_t &size);
    bool isRemoteTheSame(const std::string &dataToCompare);
    void setRemoteMemory(const std::string& dataToSet);

    virtual ~RemoteMemory_Helper();

private:
    HddlUnite::RemoteMemory::Ptr _remoteMemory = nullptr;

};

//------------------------------------------------------------------------------
inline RemoteMemory_Helper::~RemoteMemory_Helper() {
    destroyRemoteMemory();
}

inline HddlUnite::RemoteMemory::Ptr RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID &id,
                                                                const InferenceEngine::TensorDesc& tensorDesc) {
    const size_t size = InferenceEngine::details::product(
            tensorDesc.getDims().begin(), tensorDesc.getDims().end());
    return allocateRemoteMemory(id, size);
}

inline HddlUnite::RemoteMemory::Ptr
RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID &id, const size_t &size) {
    if (_remoteMemory != nullptr) {
        std::cerr << "Memory already allocated!" << std::endl;
        return 0;
    }

    HddlUnite::WorkloadContext::Ptr context = HddlUnite::queryWorkloadContext(id);
    if (context == nullptr) {
        std::cerr << "Incorrect workload id!" << std::endl;
        return 0;
    }

    _remoteMemory = HddlUnite::allocate(*context, size);
    if (_remoteMemory == nullptr) {
        return 0;
    }

    return _remoteMemory;
}

inline void RemoteMemory_Helper::destroyRemoteMemory() {
    _remoteMemory = nullptr;
}

inline std::string RemoteMemory_Helper::getRemoteMemory(const size_t &size) {
    std::vector<char> tempBuffer;
    tempBuffer.resize(size);
    auto retCode = _remoteMemory->syncFromDevice(tempBuffer.data(), size);
    if (retCode != HDDL_OK) {
        std::cerr << "[ERROR] Failed to sync memory from device!" << std::endl;
        return "";
    }
    return std::string(tempBuffer.begin(), tempBuffer.end());
}

inline bool RemoteMemory_Helper::isRemoteTheSame(const std::string &dataToCompare) {
    const size_t size = dataToCompare.size();
    const std::string remoteMemory = getRemoteMemory(size);
    if (dataToCompare != remoteMemory) {
        std::cout << "Handle: " << _remoteMemory->getDmaBufFd() << " Remote memory " << remoteMemory
                     << " != local memory " << dataToCompare << std::endl;
        return false;
    }
    return true;
}

inline void RemoteMemory_Helper::setRemoteMemory(const std::string& dataToSet) {
    _remoteMemory->syncToDevice(dataToSet.data(), dataToSet.size());
}
