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
#include <hddl2_helper.h>

#include "hddl2/hddl2_params.hpp"
#include "hddl2_exceptions.h"
#include "hddl2_executor.h"
// Subplugin
#include "video_workload_device.h"

namespace vpux {
namespace HDDL2 {
namespace IE = InferenceEngine;

ParsedContextParams::ParsedContextParams(const InferenceEngine::ParamMap& paramMap): _paramMap(paramMap) {
    // TODO Add trace logging
    if (_paramMap.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Param map for context is empty.";
    }
    // Get workload id and based on it get HddlUniteContext
    if (_paramMap.find(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)) == _paramMap.end()) {
        IE_THROW() << PARAMS_ERROR_str << "Param map does not contain workload id information";
    }
    try {
        _workloadId = _paramMap.at(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)).as<uint64_t>();
    } catch (...) {
        IE_THROW() << PARAMS_ERROR_str << "ParsedContextParams: Incorrect type of WORKLOAD_CONTEXT_ID.";
    }
}

InferenceEngine::ParamMap ParsedContextParams::getParamMap() const {
    return _paramMap;
}

WorkloadID ParsedContextParams::getWorkloadId() const {
    return _workloadId;
}

//------------------------------------------------------------------------------
VideoWorkloadDevice::VideoWorkloadDevice(const InferenceEngine::ParamMap& paramMap, const VPUXConfig& config)
        : _contextParams(paramMap) {
    // TODO Create logger for context device
    _workloadContext = HddlUnite::queryWorkloadContext(_contextParams.getWorkloadId());
    if (_workloadContext == nullptr) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Context is not found.";
    }
    if (_workloadContext->getDevice() == nullptr) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Device from context not found.";
    }
    _name = _workloadContext->getDevice()->getName();
    _allocatorPtr = std::make_shared<vpu::HDDL2Plugin::HDDL2RemoteAllocator>(_workloadContext, config.logLevel());
}

vpux::Executor::Ptr VideoWorkloadDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                        const VPUXConfig& config) {
    return vpux::HDDL2::HDDL2Executor::prepareExecutor(networkDescription, config, _allocatorPtr, _workloadContext);
}

std::shared_ptr<Allocator> VideoWorkloadDevice::getAllocator(const InferenceEngine::ParamMap& paramMap) const {
    try {
        // VideoWorkload allocator only suitable for HddlUnite::RemoteMemory. Will throw, if not found.
        const auto remoteMemory = vpux::HDDL2::getRemoteMemoryFromParams(paramMap);
        if (remoteMemory != nullptr) {
            return _allocatorPtr;
        }
    } catch (...) {
        // TODO Add log message that allocator for such params not found.
    }
    IE_THROW() << "VideoWorkloadDevice: Appropriate allocator for provided params cannot be found.";
}
}  // namespace HDDL2
}  // namespace vpux
