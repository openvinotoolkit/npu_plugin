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

#include "RemoteMemory.h"
#include "WorkloadContext.h"
#include "hddl2_remote_allocator.h"
#include "hddl2_remote_context.h"
#include "ie_remote_context.hpp"

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
//      class HDDL2BlobParams
//------------------------------------------------------------------------------
class HDDL2BlobParams {
public:
    explicit HDDL2BlobParams(const InferenceEngine::ParamMap& paramMap, const HDDL2Config& config);

    InferenceEngine::ParamMap getParamMap() const { return _paramMap; };
    RemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; };
    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; };
    std::shared_ptr<InferenceEngine::ROI> getROIPtr() const { return _roiPtr; };

protected:
    InferenceEngine::ParamMap _paramMap;
    RemoteMemoryFD _remoteMemoryFd;
    InferenceEngine::ColorFormat _colorFormat;
    std::shared_ptr<InferenceEngine::ROI> _roiPtr;
    const Logger::Ptr _logger;
};

//------------------------------------------------------------------------------
//      class HDDL2RemoteBlob
//------------------------------------------------------------------------------
class HDDL2RemoteBlob : public InferenceEngine::RemoteBlob {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteBlob>;

    explicit HDDL2RemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const HDDL2RemoteContext::Ptr& contextPtr,
        const InferenceEngine::ParamMap& params, const HDDL2Config& config);
    ~HDDL2RemoteBlob() override = default;

    /**
     * @details Since Remote blob just wrap remote memory, allocation is not required
     */
    void allocate() noexcept override {};

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

    InferenceEngine::ParamMap getParams() const override { return _params.getParamMap(); };

    std::string getDeviceName() const noexcept override;

    RemoteMemoryFD getRemoteMemoryFD() const { return _remoteMemoryFd; };

    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; };

    std::shared_ptr<InferenceEngine::ROI> getROIPtr() const { return _roiPtr; };

    size_t size() const noexcept override;

    size_t byteSize() const noexcept override;

protected:
    void* _memoryHandle = nullptr;

    const HDDL2BlobParams _params;
    std::weak_ptr<HDDL2RemoteContext> _remoteContextPtr;
    std::shared_ptr<InferenceEngine::IAllocator> _allocatorPtr = nullptr;

    const HDDL2Config& _config;
    const RemoteMemoryFD _remoteMemoryFd;
    const InferenceEngine::ColorFormat _colorFormat;
    const std::shared_ptr<InferenceEngine::ROI> _roiPtr;
    const Logger::Ptr _logger;

    void* getHandle() const noexcept override;
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
