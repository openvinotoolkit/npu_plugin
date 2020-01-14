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

#include "WorkloadContext.h"

//------------------------------------------------------------------------------
//      class HDDL2_WorkloadContext_Helper
//------------------------------------------------------------------------------
class HDDL2_WorkloadContext_Helper {
public:
    HDDL2_WorkloadContext_Helper();
    ~HDDL2_WorkloadContext_Helper();

    static WorkloadID createAndRegisterWorkloadContext();
    static void destroyHddlUniteContext(const WorkloadID& id);
    static bool isValidWorkloadContext(const WorkloadID& id);

    WorkloadID getWorkloadId();
protected:
    WorkloadID _workloadId;
};

//------------------------------------------------------------------------------
//      class HDDL2_WorkloadContext_Helper Implementation
//------------------------------------------------------------------------------
inline HDDL2_WorkloadContext_Helper::HDDL2_WorkloadContext_Helper() {
    _workloadId = createAndRegisterWorkloadContext();
}

inline HDDL2_WorkloadContext_Helper::~HDDL2_WorkloadContext_Helper() {
    destroyHddlUniteContext(_workloadId);
}

inline WorkloadID HDDL2_WorkloadContext_Helper::createAndRegisterWorkloadContext() {
    WorkloadID id = -1;
    auto context = HddlUnite::createWorkloadContext();

    auto ret = context->setContext(id);
    if (ret != HDDL_OK) {
        throw std::runtime_error("Error: WorkloadContext set context failed\"");
    }
    ret = registerWorkloadContext(context);
    if (ret != HDDL_OK) {
        throw std::runtime_error("Error: WorkloadContext register on WorkloadCache failed");
    }

    return id;
}

inline void HDDL2_WorkloadContext_Helper::destroyHddlUniteContext(const WorkloadID &id) {
    HddlUnite::unregisterWorkloadContext(id);
}

inline WorkloadID HDDL2_WorkloadContext_Helper::getWorkloadId() {
    return _workloadId;
}

inline bool HDDL2_WorkloadContext_Helper::isValidWorkloadContext(const WorkloadID &id) {
    HddlUnite::WorkloadContext::Ptr workloadContext = HddlUnite::queryWorkloadContext(id);
    return workloadContext != nullptr;
}
