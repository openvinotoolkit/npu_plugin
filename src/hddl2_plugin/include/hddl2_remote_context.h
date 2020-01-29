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

#include <memory>
#include <string>

#include "WorkloadContext.h"
#include "hddl2_remote_allocator.h"
#include "ie_remote_context.hpp"

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
//      class HDDL2ContextParams
//------------------------------------------------------------------------------
class HDDL2ContextParams {
public:
    explicit HDDL2ContextParams(const InferenceEngine::ParamMap& paramMap);

    InferenceEngine::ParamMap getParamMap() const;
    WorkloadID getWorkloadId() const;

protected:
    InferenceEngine::ParamMap _paramMap;
    WorkloadID _workloadId;
};

//------------------------------------------------------------------------------
//      class HDDL2RemoteContext
//------------------------------------------------------------------------------
class HDDL2RemoteContext :
    public InferenceEngine::RemoteContext,
    public std::enable_shared_from_this<HDDL2RemoteContext> {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteContext>;

    /**
     * @brief Constructor with parameters, to initialize from workload id
     */
    explicit HDDL2RemoteContext(const InferenceEngine::ParamMap& paramMap);

    /**
     * @brief CreateBlob provide ability to create RemoteBlob from remote memory fd
     */
    InferenceEngine::RemoteBlob::Ptr CreateBlob(
        const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) noexcept override;

    /**
     * @brief Provide device name attached to current context.
     * Format: {plugin prefix}.{device name}
     */
    std::string getDeviceName() const noexcept override;

    InferenceEngine::ParamMap getParams() const override;
    HDDL2RemoteAllocator::Ptr getAllocator();
    HddlUnite::WorkloadContext::Ptr getHddlUniteWorkloadContext() const;

protected:
    HDDL2ContextParams _contextParams;
    HDDL2RemoteAllocator::Ptr _allocatorPtr = nullptr;

    HddlUnite::WorkloadContext::Ptr _workloadContext = nullptr;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
