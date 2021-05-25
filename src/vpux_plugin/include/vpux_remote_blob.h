//
// Copyright 2020 Intel Corporation.
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
    ~VPUXRemoteBlob() override;

    /** @details Since Remote blob just wraps remote memory, allocation is not required */
    void allocate() noexcept override {
    }

    /** @details Since Remote blob just wraps remote memory, deallocation does nothing */
    bool deallocate() noexcept override {
        return false;
    }
    InferenceEngine::LockedMemory<void> buffer() noexcept override;

    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override;

    InferenceEngine::LockedMemory<void> rwmap() noexcept override;

    InferenceEngine::LockedMemory<const void> rmap() const noexcept override;

    InferenceEngine::LockedMemory<void> wmap() noexcept override;

    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override;

    std::string getDeviceName() const noexcept override;

    InferenceEngine::Blob::Ptr createROI(const InferenceEngine::ROI& regionOfInterest) const override;

    InferenceEngine::ParamMap getParams() const override {
        return _parsedParams.getParamMap();
    }

    void updateColorFormat(const InferenceEngine::ColorFormat colorFormat);

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

    InferenceEngine::TensorDesc getOriginalTensorDesc() const {
        return _originalTensorDesc;
    }
    void* getHandle() const noexcept override {
        return _memoryHandle;
    }
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

}  // namespace vpux
