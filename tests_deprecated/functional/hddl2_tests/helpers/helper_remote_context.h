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

#include "Inference.h"
#include <HddlUnite.h>
#include <WorkloadContext.h>
#include <hddl2_helpers/helper_workload_context.h>

// [Track number: E#12122]
// TODO Remove this header after removing HDDL2 deprecated parameters in future releases
#include "hddl2/hddl2_params.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_ie_core.h"

//------------------------------------------------------------------------------
//      class Remote_Context_Helper
//------------------------------------------------------------------------------
class Remote_Context_Helper: public IE_Core_Helper {
public:
    Remote_Context_Helper();

    InferenceEngine::RemoteContext::Ptr remoteContextPtr = nullptr;
    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);

    WorkloadID getWorkloadId() const;

protected:
    WorkloadContext_Helper _workloadContext;
};

//------------------------------------------------------------------------------
//      class Remote_Context_Helper Implementation
//------------------------------------------------------------------------------
inline Remote_Context_Helper::Remote_Context_Helper() {
    auto params = wrapWorkloadIdToMap(_workloadContext.getWorkloadId());
    remoteContextPtr = ie.CreateContext(pluginName, params);
}

inline InferenceEngine::ParamMap
Remote_Context_Helper::wrapWorkloadIdToMap(const WorkloadID &id) {
    // [Track number: E#12122]
    // TODO Remove HDDL2_PARAM_KEY part after removing deprecated HDDL2 parameters in future releases
    if (std::rand()%2) {
        return {{InferenceEngine::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
    }
    return {{InferenceEngine::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID), id}};
}

inline WorkloadID Remote_Context_Helper::getWorkloadId() const {
    return _workloadContext.getWorkloadId();
}
