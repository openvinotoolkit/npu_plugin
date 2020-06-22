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

#include "kmb_remote_context.h"
#include "ie_remote_context.hpp"

namespace vpu {
namespace KmbPlugin {

class KmbBlobParams {
public:
    explicit KmbBlobParams(const InferenceEngine::ParamMap& paramMap, const KmbConfig& config);

    InferenceEngine::ParamMap getParamMap() const { return _paramMap; }
    KmbRemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; }
    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; }

protected:
    InferenceEngine::ParamMap _paramMap;
    KmbRemoteMemoryFD _remoteMemoryFd;
    InferenceEngine::ColorFormat _colorFormat;
    const Logger::Ptr _logger;
};

class KmbRemoteBlob : public InferenceEngine::RemoteBlob {
public:
    using Ptr = std::shared_ptr<KmbRemoteBlob>;

    explicit KmbRemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const KmbRemoteContext::Ptr& contextPtr,
        const InferenceEngine::ParamMap& params, const KmbConfig& config);
    ~KmbRemoteBlob() override = default;

    /**
     * @details Since Remote blob just wrap remote memory, allocation is not required
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

    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; }

    size_t size() const noexcept override;

    size_t byteSize() const noexcept override;

protected:
    void* _memoryHandle = nullptr;

    const KmbBlobParams _params;
    std::weak_ptr<KmbRemoteContext> _remoteContextPtr;
    std::shared_ptr<InferenceEngine::IAllocator> _allocatorPtr = nullptr;

    const KmbConfig& _config;
    const KmbRemoteMemoryFD _remoteMemoryFd;
    const InferenceEngine::ColorFormat _colorFormat;
    const Logger::Ptr _logger;

    void* getHandle() const noexcept override;
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

} // namespace KmbPlugin
} // namespace vpu
