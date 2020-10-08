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

// Plugin
#include "hddl2_exceptions.h"
#include "hddl2_executor.h"
#include "hddl2_params.hpp"
// Subplugin
#include "subplugin/hddl2_context_device.h"

namespace vpux {
namespace HDDL2 {
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
    _workloadId = paramMap.at(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID));
    _paramMap = paramMap;
}

InferenceEngine::ParamMap HDDL2ContextParams::getParamMap() const { return _paramMap; }

WorkloadID HDDL2ContextParams::getWorkloadId() const { return _workloadId; }

//------------------------------------------------------------------------------
HDDLUniteContextDevice::HDDLUniteContextDevice(const InferenceEngine::ParamMap& paramMap, const VPUXConfig& config)
    : _contextParams(paramMap) {
    // TODO Create logger for context device
    _workloadContext = HddlUnite::queryWorkloadContext(_contextParams.getWorkloadId());
    if (_workloadContext == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Context is not found.";
    }
    if (_workloadContext->getDevice() == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Device from context not found.";
    }
    _name = _workloadContext->getDevice()->getName();
    _allocatorPtr = std::make_shared<vpu::HDDL2Plugin::HDDL2RemoteAllocator>(_workloadContext, config.logLevel());
}

vpux::Executor::Ptr HDDLUniteContextDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    return vpux::HDDL2::HDDL2Executor::prepareExecutor(networkDescription, config, _workloadContext);
}
}  // namespace HDDL2
}  // namespace vpux
