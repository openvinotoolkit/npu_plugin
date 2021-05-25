//
// Copyright 2019 Intel Corporation.
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

#include "hddl2_helpers/helper_workload_context.h"
#include "helper_hddl2_backend.h"
#include "vpux/vpux_plugin_params.hpp"
#include "ie_remote_context.hpp"
#include "vpux_remote_context.h"

namespace vpux {
namespace hddl2 {

//------------------------------------------------------------------------------
class RemoteContext_Helper {
public:
    using Ptr = std::shared_ptr<RemoteContext_Helper>;
    RemoteContext_Helper();
    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);
    WorkloadID getWorkloadId() const;
    HddlUnite::WorkloadContext::Ptr getWorkloadContext();

    vpux::VPUXRemoteContext::Ptr remoteContextPtr = nullptr;
    const vpux::VPUXConfig config;

protected:
    // TODO Use stub instead of creating "default" _workloadContext
    WorkloadContext_Helper _workloadContextHelper;
};

//------------------------------------------------------------------------------
inline RemoteContext_Helper::RemoteContext_Helper() {
    vpux::HDDL2Backend_Helper _backendHelper;

    auto param = wrapWorkloadIdToMap(_workloadContextHelper.getWorkloadId());
    auto device = _backendHelper.getDevice(param);
    remoteContextPtr = std::make_shared<vpux::VPUXRemoteContext>(device, param);
}

inline InferenceEngine::ParamMap
RemoteContext_Helper::wrapWorkloadIdToMap(const WorkloadID &id) {
    return {{InferenceEngine::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}

inline WorkloadID RemoteContext_Helper::getWorkloadId() const {
    return _workloadContextHelper.getWorkloadId();
}

inline HddlUnite::WorkloadContext::Ptr RemoteContext_Helper::getWorkloadContext() {
    return _workloadContextHelper.getWorkloadContext();
}

}
}
