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

#include <hddl2_remote_blob.h>
#include <hddl2_remote_context.h>

#include <cpp_interfaces/exception2status.hpp>
#include <memory>
#include <string>

#include "hddl2_exceptions.h"
#include "hddl2_params.hpp"

using namespace vpu::HDDL2Plugin;

namespace IE = InferenceEngine;

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
    _workloadId = workload_ctx_iter->second.as<RemoteMemoryFD>();

    _paramMap = paramMap;
}

InferenceEngine::ParamMap HDDL2ContextParams::getParamMap() const { return _paramMap; }

WorkloadID HDDL2ContextParams::getWorkloadId() const { return _workloadId; }

//------------------------------------------------------------------------------
HDDL2RemoteContext::HDDL2RemoteContext(const InferenceEngine::ParamMap& paramMap, const vpu::HDDL2Config& config)
    : _contextParams(paramMap),
      _config(config),
      _logger(std::make_shared<Logger>("HDDL2RemoteContext", config.logLevel(), consoleOutput())) {
    _workloadContext = HddlUnite::queryWorkloadContext(_contextParams.getWorkloadId());
    if (_workloadContext == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "context is not found";
    }
    _allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(_workloadContext, _config);
}

IE::RemoteBlob::Ptr HDDL2RemoteContext::CreateBlob(
    const IE::TensorDesc& tensorDesc, const IE::ParamMap& params) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger->warning("Please use smart ptr to context instead of instance of class\n");
        return nullptr;
    }
    try {
        return std::make_shared<HDDL2RemoteBlob>(tensorDesc, shared_from_this(), params, _config);
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory fd is correct.\nError: %s\n",
            ex.what());
        return nullptr;
    }
}

std::string HDDL2RemoteContext::getDeviceName() const noexcept {
    if (_workloadContext == nullptr) {
        return "";
    }
    return "VPUX." + _workloadContext->getDevice()->getName();
}

IE::ParamMap HDDL2RemoteContext::getParams() const { return _contextParams.getParamMap(); }

HDDL2RemoteAllocator::Ptr HDDL2RemoteContext::getAllocator() { return _allocatorPtr; }

HddlUnite::WorkloadContext::Ptr HDDL2RemoteContext::getHddlUniteWorkloadContext() const { return _workloadContext; }
