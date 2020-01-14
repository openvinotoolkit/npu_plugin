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

#include "hddl2_helpers/hddl2_workload_context.h"
#include "hddl2_remote_context.h"
#include "hddl2_params.hpp"

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
//      class HDDL2_With_RemoteContext_Helper
//------------------------------------------------------------------------------
class HDDL2_With_RemoteContext_Helper {
public:
    HDDL2_With_RemoteContext_Helper();

    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);

    HDDL2RemoteContext::Ptr remoteContextPtr = nullptr;

protected:
    // TODO Use stub instead of creating "default" workloadContext
    HDDL2_WorkloadContext_Helper workloadContext;
};

//------------------------------------------------------------------------------
//      class HDDL2_With_RemoteContext_Helper Implementation
//------------------------------------------------------------------------------
inline HDDL2_With_RemoteContext_Helper::HDDL2_With_RemoteContext_Helper() {
    auto param = wrapWorkloadIdToMap(workloadContext.getWorkloadId());
    remoteContextPtr = std::make_shared<HDDL2RemoteContext>(param);
}

inline InferenceEngine::ParamMap
HDDL2_With_RemoteContext_Helper::wrapWorkloadIdToMap(const WorkloadID &id) {
    return {{InferenceEngine::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}

}
}
