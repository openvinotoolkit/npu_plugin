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

// Plugin
#include <hddl2_helper.h>
#include <device_helpers.hpp>

#include "hddl2_exceptions.h"
#include "hddl2_executor.h"
// [Track number: E#12122]
// TODO Remove this header after removing HDDL2 deprecated parameters in future releases
#include "hddl2/hddl2_params.hpp"
#include "vpux/vpux_plugin_params.hpp"
// Subplugin
#include "video_workload_device.h"

namespace vpux {
namespace hddl2 {
namespace IE = InferenceEngine;

ParsedContextParams::ParsedContextParams(const InferenceEngine::ParamMap& paramMap): _paramMap(paramMap) {
    // TODO Add trace logging
    if (_paramMap.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Param map for context is empty.";
    }
    // Get workload id and based on it get HddlUniteContext

    // [Track number: E#12122]
    // TODO Remove this deprecated part after removing HDDL2 deprecated parameters in future releases
    if (_paramMap.find(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)) != _paramMap.end()) {
        try {
            _workloadId = _paramMap.at(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)).as<uint64_t>();
        } catch (...) {
            IE_THROW() << PARAMS_ERROR_str << "ParsedContextParams: Incorrect type of WORKLOAD_CONTEXT_ID.";
        }
    } else {
        if (_paramMap.find(IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID)) == _paramMap.end()) {
            IE_THROW() << PARAMS_ERROR_str << "Param map does not contain workload id information";
        }
        try {
            _workloadId = _paramMap.at(IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID)).as<uint64_t>();
        } catch (...) {
            IE_THROW() << PARAMS_ERROR_str << "ParsedContextParams: Incorrect type of WORKLOAD_CONTEXT_ID.";
        }
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
    const auto swDeviceId = _workloadContext->getDevice()->getSwDeviceId();
    _name = utils::getDeviceNameBySwDeviceId(swDeviceId);
    _allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(_workloadContext, config.logLevel());
}

vpux::Executor::Ptr VideoWorkloadDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                        const VPUXConfig& config) {
    return HDDL2Executor::prepareExecutor(networkDescription, config, _allocatorPtr, _workloadContext);
}

std::shared_ptr<Allocator> VideoWorkloadDevice::getAllocator(const InferenceEngine::ParamMap& paramMap) const {
    try {
        // VideoWorkload allocator is suitable only for HddlUnite::RemoteMemory. Will throw, if it was not found.
        const auto remoteMemoryFD = getRemoteMemoryFDFromParams(paramMap);
        if (remoteMemoryFD >= 0) {
            return _allocatorPtr;
        }
    } catch (...) {
        // TODO Add log message that allocator for such params not found.
    }
    IE_THROW() << "VideoWorkloadDevice: Appropriate allocator for provided params cannot be found.";
}
}  // namespace hddl2
}  // namespace vpux
