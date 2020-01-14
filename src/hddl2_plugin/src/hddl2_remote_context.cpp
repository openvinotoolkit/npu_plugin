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

#include <hddl2_remote_context.h>

#include <cpp_interfaces/exception2status.hpp>
#include <memory>
#include <string>

#include "hddl2_exceptions.h"
#include "hddl2_params.hpp"

using namespace vpu::HDDL2Plugin;

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2ContextParams Implementation
//------------------------------------------------------------------------------
HDDL2ContextParams::HDDL2ContextParams(const InferenceEngine::ParamMap& paramMap) {
    if (paramMap.empty()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Param map for context is empty.";
    }
    // Get workload id and based on it get HddlUniteContext
    auto workload_ctx_iter = paramMap.find(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID));
    if (workload_ctx_iter == paramMap.end()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Param map does not contain workload id information";
    }
    _workloadId = workload_ctx_iter->second.as<uint64_t>();
    _paramMap = paramMap;
}

InferenceEngine::ParamMap HDDL2ContextParams::getParamMap() const { return _paramMap; }

WorkloadID HDDL2ContextParams::getWorkloadId() const { return _workloadId; }

//------------------------------------------------------------------------------
//      class HDDL2RemoteContext Implementation
//------------------------------------------------------------------------------
HDDL2RemoteContext::HDDL2RemoteContext(const InferenceEngine::ParamMap& paramMap): _contextParams(paramMap) {
    _workloadContext = HddlUnite::queryWorkloadContext(_contextParams.getWorkloadId());
    if (_workloadContext == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "context is not found";
    }
}

IE::RemoteBlob::Ptr HDDL2RemoteContext::CreateBlob(
    const IE::TensorDesc& tensorDesc, const IE::ParamMap& params) noexcept {
    UNUSED(tensorDesc);
    UNUSED(params);
    return nullptr;
}

std::string HDDL2RemoteContext::getDeviceName() const noexcept {
    if (_workloadContext == nullptr) {
        return "";
    }
    return "HDDL2." + _workloadContext->getDevice()->getName();
}

IE::ParamMap HDDL2RemoteContext::getParams() const { return _contextParams.getParamMap(); }

HDDL2RemoteAllocator::Ptr HDDL2RemoteContext::getAllocator() { return _allocatorPtr; }

HddlUnite::WorkloadContext::Ptr HDDL2RemoteContext::getHddlUniteWorkloadContext() const { return _workloadContext; }
