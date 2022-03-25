//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
#include <ie_core.hpp>
#include "functional_test_utils/plugin_cache.hpp"

//------------------------------------------------------------------------------
//      class Remote_Context_Helper
//------------------------------------------------------------------------------
class Remote_Context_Helper {
public:
    Remote_Context_Helper();

    InferenceEngine::RemoteContext::Ptr remoteContextPtr = nullptr;
    static InferenceEngine::ParamMap wrapWorkloadIdToMap(const WorkloadID &id);

    WorkloadID getWorkloadId() const;

protected:
    WorkloadContext_Helper _workloadContext;
    std::string pluginName;
};

//------------------------------------------------------------------------------
//      class Remote_Context_Helper Implementation
//------------------------------------------------------------------------------
inline Remote_Context_Helper::Remote_Context_Helper() {
    pluginName = std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "VPUX";
    auto params = wrapWorkloadIdToMap(_workloadContext.getWorkloadId());
    remoteContextPtr = PluginCache::get().ie()->CreateContext(pluginName, params);
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
