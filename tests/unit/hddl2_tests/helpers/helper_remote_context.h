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

#pragma once

#include "ie_remote_context.hpp"

#include "hddl2_helpers/helper_workload_context.h"
#include "hddl2_remote_context.h"
#include "hddl2_params.hpp"

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
//      class RemoteContext_Helper
//------------------------------------------------------------------------------
class RemoteContext_Helper {
public:
    RemoteContext_Helper();
    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);
    WorkloadID getWorkloadId() const;

    HDDL2RemoteContext::Ptr remoteContextPtr = nullptr;
    const vpu::HDDL2Config config;

protected:
    // TODO Use stub instead of creating "default" _workloadContext
    WorkloadContext_Helper _workloadContext;
};

//------------------------------------------------------------------------------
//      class RemoteContext_Helper Implementation
//------------------------------------------------------------------------------
inline RemoteContext_Helper::RemoteContext_Helper() {
    auto param = wrapWorkloadIdToMap(_workloadContext.getWorkloadId());
    remoteContextPtr = std::make_shared<HDDL2RemoteContext>(param, config);
}

inline InferenceEngine::ParamMap
RemoteContext_Helper::wrapWorkloadIdToMap(const WorkloadID &id) {
    return {{InferenceEngine::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}

inline WorkloadID RemoteContext_Helper::getWorkloadId() const {
    return _workloadContext.getWorkloadId();
}

}
}
