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

    int allocateRemoteMemory(const WorkloadID& id, const size_t width, const size_t height,
        const size_t widthStride, const size_t heightStride, const void* data = nullptr,
        const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);
    int allocateRemoteMemory(const WorkloadID& id, const size_t size, const void* data = nullptr,
        const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);
    int allocateRemoteMemory(const WorkloadID& id, const InferenceEngine::TensorDesc& tensorDesc,
        const void* data = nullptr, const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);

    HddlUnite::RemoteMemory::Ptr allocateRemoteMemoryPtr(const WorkloadID& id, const size_t width, const size_t height,
        const size_t widthStride, const size_t heightStride, const void* data = nullptr,
        const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);
    HddlUnite::RemoteMemory::Ptr allocateRemoteMemoryPtr(const WorkloadID& id, const size_t size, const void* data = nullptr,
        const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);
    HddlUnite::RemoteMemory::Ptr allocateRemoteMemoryPtr(const WorkloadID& id, const InferenceEngine::TensorDesc& tensorDesc,
        const void* data = nullptr, const HddlUnite::eRemoteMemoryFormat format = HddlUnite::eRemoteMemoryFormat::NONE);

    void destroyRemoteMemory();

    void clearRemoteMemory();
    std::string getRemoteMemory(const size_t size);
    HddlUnite::RemoteMemory::Ptr getRemoteMemoryPtr() {return _remoteMemory;}
    bool isRemoteTheSame(const std::string& dataToCompare);
    void setRemoteMemory(const std::string& dataToSet);

    virtual ~RemoteMemory_Helper();

private:
    HddlUnite::RemoteMemory::Ptr _remoteMemory = nullptr;

};

//------------------------------------------------------------------------------
inline RemoteMemory_Helper::~RemoteMemory_Helper() {
    destroyRemoteMemory();
}

inline int RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID& id, const size_t width, const size_t height,
    const size_t widthStride, const size_t heightStride, const void* data, const HddlUnite::eRemoteMemoryFormat format) {
    return allocateRemoteMemoryPtr(id, width, height, widthStride, heightStride, data, format)->getDmaBufFd();
}

inline int RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID& id,
       const InferenceEngine::TensorDesc& tensorDesc, const void* data, const HddlUnite::eRemoteMemoryFormat format) {
    return allocateRemoteMemoryPtr(id, tensorDesc, data, format)->getDmaBufFd();
}

inline int
RemoteMemory_Helper::allocateRemoteMemory(const WorkloadID& id, const size_t size, const void* data,
    const HddlUnite::eRemoteMemoryFormat format) {
    return allocateRemoteMemoryPtr(id, size ,data, format)->getDmaBufFd();
}

inline HddlUnite::RemoteMemory::Ptr RemoteMemory_Helper::allocateRemoteMemoryPtr(const WorkloadID& id, const size_t width, const size_t height,
    const size_t widthStride, const size_t heightStride, const void* data, const HddlUnite::eRemoteMemoryFormat format) {
    if (_remoteMemory != nullptr) {
       IE_THROW()  << "Remote memory is already allocated.";
    }

    HddlUnite::WorkloadContext::Ptr context = HddlUnite::queryWorkloadContext(id);
    if (context == nullptr) {
        IE_THROW() << "Incorrect workload id.";
    }

    _remoteMemory = std::make_shared<HddlUnite::RemoteMemory>(*context, HddlUnite::RemoteMemoryDesc(width, height, widthStride, heightStride, format));
    if (_remoteMemory == nullptr) {
        IE_THROW() << "Failed to allocate remote memory.";
    }

    if (data != nullptr) {
        const size_t dataSize = _remoteMemory->getMemoryDesc().getDataSize();
        if (_remoteMemory->syncToDevice(data, dataSize) != HDDL_OK) {
            IE_THROW() << "Failed to sync memory to device.";
        }
    }

    return _remoteMemory;
}

inline HddlUnite::RemoteMemory::Ptr RemoteMemory_Helper::allocateRemoteMemoryPtr(const WorkloadID& id,
       const InferenceEngine::TensorDesc& tensorDesc, const void* data, const HddlUnite::eRemoteMemoryFormat format) {
    const auto& dims = tensorDesc.getDims();
    if (dims.size() != 4) {
    const size_t size = InferenceEngine::details::product(
        tensorDesc.getDims().begin(), tensorDesc.getDims().end());
        return allocateRemoteMemoryPtr(id, size, 1, size, 1, data, format);
    }

    const bool isNV12 = (format == HddlUnite::eRemoteMemoryFormat::NV12);
    const int H_index = 2;
    const int W_index = 3;

    const auto& blockingDesc = tensorDesc.getBlockingDesc();
    const auto& strides = blockingDesc.getStrides();
        if (strides.empty()) {
            IE_THROW() << "Strides information is not provided.";
        }

    // Define strides and dimensions. Only NCHW/NHWC orders/layouts are supported. NV12 always has NHWC order
    const bool isNCHW = isNV12 ? false : (tensorDesc.getLayout() == InferenceEngine::Layout::NCHW);
    const size_t mWidth = dims[W_index];
    const size_t mHeight = dims[H_index];
    const size_t mWidthStride = strides[isNCHW ? 2 : 1];
    const size_t mHeightStride = strides[isNCHW ? 1 : 0] / mWidthStride;
    return allocateRemoteMemoryPtr(id, mWidth, mHeight, mWidthStride, mHeightStride, data, format);
}

inline HddlUnite::RemoteMemory::Ptr
RemoteMemory_Helper::allocateRemoteMemoryPtr(const WorkloadID& id, const size_t size, const void* data,
    const HddlUnite::eRemoteMemoryFormat format) {
    return allocateRemoteMemoryPtr(id, size, 1, size, 1, data, format);
}

inline void RemoteMemory_Helper::destroyRemoteMemory() {
    _remoteMemory = nullptr;
}

inline void RemoteMemory_Helper::clearRemoteMemory() {
    if (_remoteMemory == nullptr) {
        std::cerr << "[ERROR] Failed to clear remote memory - null pointer!" << std::endl;
    }

    const auto size = _remoteMemory->getMemoryDesc().getDataSize();
    const std::vector<char> zeroData(size, 0);
    auto retCode = _remoteMemory->syncToDevice(zeroData.data(), zeroData.size());
    if (retCode != HDDL_OK) {
        std::cerr << "[ERROR] Failed to clear remote memory - sync memory to device!" << std::endl;
    }
}

inline std::string RemoteMemory_Helper::getRemoteMemory(const size_t size) {
    std::vector<char> tempBuffer;
    tempBuffer.resize(size);
    auto retCode = _remoteMemory->syncFromDevice(tempBuffer.data(), size);
    if (retCode != HDDL_OK) {
        std::cerr << "[ERROR] Failed to sync memory from device!" << std::endl;
        return "";
    }
    return std::string(tempBuffer.begin(), tempBuffer.end());
}

inline bool RemoteMemory_Helper::isRemoteTheSame(const std::string& dataToCompare) {
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
