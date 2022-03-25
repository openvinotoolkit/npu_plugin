//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "hddl2_helpers/helper_workload_context.h"
#include "helper_hddl2_backend.h"
#include "ie_remote_context.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_remote_context.h"

namespace vpux {
namespace hddl2 {

//------------------------------------------------------------------------------
class RemoteContext_Helper {
public:
    using Ptr = std::shared_ptr<RemoteContext_Helper>;
    RemoteContext_Helper();
    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID& id);
    WorkloadID getWorkloadId() const;
    HddlUnite::WorkloadContext::Ptr getWorkloadContext();

    vpux::VPUXRemoteContext::Ptr remoteContextPtr = nullptr;

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

inline InferenceEngine::ParamMap RemoteContext_Helper::wrapWorkloadIdToMap(const WorkloadID& id) {
    return {{InferenceEngine::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}

inline WorkloadID RemoteContext_Helper::getWorkloadId() const {
    return _workloadContextHelper.getWorkloadId();
}

inline HddlUnite::WorkloadContext::Ptr RemoteContext_Helper::getWorkloadContext() {
    return _workloadContextHelper.getWorkloadContext();
}

}  // namespace hddl2
}  // namespace vpux
