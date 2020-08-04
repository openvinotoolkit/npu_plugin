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

#include <ie_blob.h>

#include <memory>
#include <string>

#include "ie_remote_context.hpp"
#include "kmb_remote_context.h"

namespace vpu {
namespace KmbPlugin {

class KmbBlobParams {
public:
    explicit KmbBlobParams(const InferenceEngine::ParamMap& paramMap, const KmbConfig& config);

    InferenceEngine::ParamMap getParamMap() const { return _paramMap; }
    KmbRemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; }
    KmbHandleParam getRemoteMemoryHandle() const { return _remoteMemoryHandle; }
    KmbOffsetParam getRemoteMemoryOffset() const { return _remoteMemoryOffset; }

protected:
    InferenceEngine::ParamMap _paramMap;
    KmbRemoteMemoryFD _remoteMemoryFd;
    const Logger::Ptr _logger;
    KmbHandleParam _remoteMemoryHandle;
    KmbOffsetParam _remoteMemoryOffset;
};

class KmbRemoteBlob : public InferenceEngine::RemoteBlob {
public:
    using Ptr = std::shared_ptr<KmbRemoteBlob>;

    explicit KmbRemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const KmbRemoteContext::Ptr& contextPtr,
        const InferenceEngine::ParamMap& params, const KmbConfig& config);
    explicit KmbRemoteBlob(const KmbRemoteBlob& origBlob, const InferenceEngine::ROI& regionOfInterest);
    ~KmbRemoteBlob() override = default;

    /**
     * @details Since Remote blob just wraps remote memory, allocation is not required
     */
    void allocate() noexcept override {}

    /**
     * @brief Deallocate local memory
     * @return True if allocation is done, False if deallocation is failed.
     */
    bool deallocate() noexcept override;

    InferenceEngine::LockedMemory<void> buffer() noexcept override;

    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override;

    InferenceEngine::LockedMemory<void> rwmap() noexcept override;

    InferenceEngine::LockedMemory<const void> rmap() const noexcept override;

    InferenceEngine::LockedMemory<void> wmap() noexcept override;

    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override;

    InferenceEngine::ParamMap getParams() const override { return _params.getParamMap(); }

    std::string getDeviceName() const noexcept override;

    KmbRemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; }

    size_t size() const noexcept override;

    size_t byteSize() const noexcept override;

    InferenceEngine::Blob::Ptr createROI(const InferenceEngine::ROI& regionOfInterest) const override;

protected:
    void* _memoryHandle = nullptr;

    const KmbBlobParams _params;
    std::weak_ptr<KmbRemoteContext> _remoteContextPtr;
    std::shared_ptr<InferenceEngine::IAllocator> _allocatorPtr = nullptr;

    const KmbConfig& _config;
    const KmbRemoteMemoryFD _remoteMemoryFd;
    const Logger::Ptr _logger;

    void* getHandle() const noexcept override;
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

}  // namespace KmbPlugin
}  // namespace vpu
