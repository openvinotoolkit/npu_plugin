//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "hddl2_helpers/helper_remote_memory.h"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_params_private_options.hpp"

namespace IE = InferenceEngine;

namespace vpux {
namespace hddl2 {

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
};

inline Allocator_Helper::Allocator_Helper(WorkloadContextPtr& workloadContextPtr) {
    allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(workloadContextPtr);
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
    IE::ParamMap paramMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), _remoteMemory->getDmaBufFd()}};
    return allocatorPtr->wrapRemoteMemory(paramMap);
}

inline Allocator_WrappedRemoteMemory_Helper::
Allocator_WrappedRemoteMemory_Helper(WorkloadContextPtr& workloadContextPtr)
                                        : Allocator_Helper(workloadContextPtr) {
    WorkloadID workloadId = workloadContextPtr->getWorkloadContextID();
    _remoteMemoryHelper.allocateRemoteMemory(workloadId, defaultSizeToAllocate);
    _remoteMemory = _remoteMemoryHelper.getRemoteMemoryPtr();
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

}  // namespace hddl2
}  // namespace vpux
