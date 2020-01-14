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

#include "Inference.h"
#include <HddlUnite.h>
#include <WorkloadContext.h>

#include "hddl2_params.hpp"
#include "hddl2_ie_core.h"

//------------------------------------------------------------------------------
//      class HDDL2_Remote_Context_Helper
//------------------------------------------------------------------------------
class HDDL2_Remote_Context_Helper: public HDDL2_IE_Core_Helper {
public:
    HDDL2_Remote_Context_Helper();

    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);
    InferenceEngine::RemoteContext::Ptr remoteContextPtr = nullptr;

protected:
    HDDL2_WorkloadContext_Helper workloadContext;
};

//------------------------------------------------------------------------------
//      class HDDL2_Remote_Context_Helper Implementation
//------------------------------------------------------------------------------
inline HDDL2_Remote_Context_Helper::HDDL2_Remote_Context_Helper() {
    auto params = wrapWorkloadIdToMap(workloadContext.getWorkloadId());
    remoteContextPtr = ie.CreateContext(pluginName, params);
}

inline InferenceEngine::ParamMap
HDDL2_Remote_Context_Helper::wrapWorkloadIdToMap(const WorkloadID &id) {
    return {{InferenceEngine::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}
