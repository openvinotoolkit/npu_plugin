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

#include "hddl2_helpers/helper_remote_memory.h"

namespace vpu {
namespace HDDL2Plugin {

using memoryHandle = void*;


//------------------------------------------------------------------------------
class Allocator_Helper {
public:
    using Ptr = std::shared_ptr<Allocator_Helper>;
    using WorkloadContextPtr = HddlUnite::WorkloadContext::Ptr;

    explicit Allocator_Helper(WorkloadContextPtr& workloadContextPtr);
    virtual ~Allocator_Helper() = default;


    /**
     * @brief Wrap allocate() and wrapRemoteMemory() methods
     */
    virtual memoryHandle createMemory(const size_t& size) = 0;
    virtual std::string getRemoteMemory(const size_t &size) = 0;

    HDDL2RemoteAllocator::Ptr allocatorPtr;
    const vpu::HDDL2Config config;
};

inline Allocator_Helper::Allocator_Helper(WorkloadContextPtr& workloadContextPtr) {
    allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(workloadContextPtr, config);
}

//------------------------------------------------------------------------------
/**
 * @brief Allocate remote memory via HddlUnite and wrap it on createMemory call
 */
class Allocator_WrappedRemoteMemory_Helper : public Allocator_Helper {
public:
    ~Allocator_WrappedRemoteMemory_Helper() override = default;
    explicit Allocator_WrappedRemoteMemory_Helper(WorkloadContextPtr& workloadContextPtr);

    memoryHandle createMemory(const size_t& size) override;
    std::string getRemoteMemory(const size_t &size) override;

    const size_t defaultSizeToAllocate = 1024 * 1024 * 4;

private:
    RemoteMemory_Helper _remoteMemoryHelper;
    HddlUnite::RemoteMemory::Ptr _remoteMemory = nullptr;
};


inline memoryHandle Allocator_WrappedRemoteMemory_Helper::createMemory(const size_t &size) {
    UNUSED(size);
    return allocatorPtr->wrapRemoteMemory(_remoteMemory);
}

inline Allocator_WrappedRemoteMemory_Helper::
Allocator_WrappedRemoteMemory_Helper(WorkloadContextPtr& workloadContextPtr)
                                        : Allocator_Helper(workloadContextPtr) {
    WorkloadID workloadId = workloadContextPtr->getWorkloadContextID();
    _remoteMemory = _remoteMemoryHelper.allocateRemoteMemory(workloadId, defaultSizeToAllocate);
}

inline std::string Allocator_WrappedRemoteMemory_Helper::getRemoteMemory(const size_t &size) {
    return _remoteMemoryHelper.getRemoteMemory(size);
}

//------------------------------------------------------------------------------
//      class Allocator_CreatedRemoteMemory_Helper
//------------------------------------------------------------------------------
/**
 * @brief HddlUnite remote memory will be allocated inside allocate() call
 */
class Allocator_CreatedRemoteMemory_Helper : public Allocator_Helper {
public:
    explicit Allocator_CreatedRemoteMemory_Helper(WorkloadContextPtr& workloadContextPtr);
    ~Allocator_CreatedRemoteMemory_Helper() override = default;
    std::string getRemoteMemory(const size_t &size) override;

private:
    memoryHandle createMemory(const size_t& size) override;
    WorkloadContextPtr _workloadContext;
};

inline memoryHandle Allocator_CreatedRemoteMemory_Helper::createMemory(const size_t &size) {
    return allocatorPtr->alloc(size);
}

inline Allocator_CreatedRemoteMemory_Helper::
Allocator_CreatedRemoteMemory_Helper(WorkloadContextPtr& workloadContextPtr) :
                                            Allocator_Helper(workloadContextPtr),
                                            _workloadContext(workloadContextPtr) {}

inline std::string Allocator_CreatedRemoteMemory_Helper::getRemoteMemory(const size_t &size) {
    UNUSED(size);
    std::cerr << "Not possible to get remote memory if InferenceEngine is owner" << std::endl;
    return "";
}

}  // namespace HDDL2Plugin
}  // namespace vpu
