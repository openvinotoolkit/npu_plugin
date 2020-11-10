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

// System
#include <memory>
#include <string>
// IE
#include "ie_blob.h"
#include "ie_remote_context.hpp"
// Plugin
#include "vpux.hpp"
#include "vpux_params.hpp"
#include "vpux_remote_context.h"

namespace vpux {

//------------------------------------------------------------------------------
class VPUXRemoteBlob final : public InferenceEngine::RemoteBlob {
public:
    using Ptr = std::shared_ptr<VPUXRemoteBlob>;
    using CPtr = std::shared_ptr<const VPUXRemoteBlob>;

    explicit VPUXRemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const VPUXRemoteContext::Ptr& contextPtr,
        const std::shared_ptr<Allocator>& allocator, const InferenceEngine::ParamMap& params,
        const vpu::LogLevel logLevel = vpu::LogLevel::None);
    ~VPUXRemoteBlob() override { VPUXRemoteBlob::deallocate(); }

    /** @details Since Remote blob just wrap remote memory, allocation is not required */
    void allocate() noexcept override {}

    bool deallocate() noexcept override;
    InferenceEngine::LockedMemory<void> buffer() noexcept override;

    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override;

    InferenceEngine::LockedMemory<void> rwmap() noexcept override;

    InferenceEngine::LockedMemory<const void> rmap() const noexcept override;

    InferenceEngine::LockedMemory<void> wmap() noexcept override;

    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override;

    std::string getDeviceName() const noexcept override;

    size_t size() const noexcept override;

    size_t byteSize() const noexcept override;

    InferenceEngine::Blob::Ptr createROI(const InferenceEngine::ROI& regionOfInterest) const override;

    InferenceEngine::ParamMap getParams() const override { return _parsedParams.getParamMap(); }

private:
    /** @brief All objects, which might be used inside backend, should be stored in paramMap */
    ParsedRemoteBlobParams _parsedParams;

    void* _memoryHandle = nullptr;

    std::weak_ptr<VPUXRemoteContext> _remoteContextPtr;
    std::shared_ptr<InferenceEngine::IAllocator> _allocatorPtr = nullptr;

    const vpu::Logger::Ptr _logger;

    /** @brief After creation ROI blob information about original tensor will be lost.
     * Since it's not possible to restore information about full tensor, keep it in separate variable */
    const InferenceEngine::TensorDesc _originalTensorDesc;

private:
    /** @details Remote ROI blob can be created only based on full frame Remote blob.
     *  IE API assume, that Remote ROI blob can be created only by using blob->createROI call on RemoteBlob
     *  This will not allow to create ROI blob using paramMap, which is not API approach. */
    explicit VPUXRemoteBlob(const VPUXRemoteBlob& origBlob, const InferenceEngine::ROI& regionOfInterest);

    InferenceEngine::TensorDesc getOriginalTensorDesc() const { return _originalTensorDesc; }
    void* getHandle() const noexcept override { return _memoryHandle; }
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

}  // namespace vpux
