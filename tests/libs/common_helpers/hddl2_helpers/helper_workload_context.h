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

#include "WorkloadContext.h"

//------------------------------------------------------------------------------
class WorkloadContext_Helper {
public:
    using Ptr = std::shared_ptr<WorkloadContext_Helper>;

    WorkloadContext_Helper();
    ~WorkloadContext_Helper();

    static WorkloadID createAndRegisterWorkloadContext();
    static void destroyHddlUniteContext(const WorkloadID& id);
    static bool isValidWorkloadContext(const WorkloadID& id);

    WorkloadID getWorkloadId() const;
    HddlUnite::WorkloadContext::Ptr getWorkloadContext() const;
protected:
    WorkloadID _workloadId;
};

//------------------------------------------------------------------------------
inline WorkloadContext_Helper::WorkloadContext_Helper() {
    _workloadId = createAndRegisterWorkloadContext();
}

inline WorkloadContext_Helper::~WorkloadContext_Helper() {
    destroyHddlUniteContext(_workloadId);
}

inline WorkloadID WorkloadContext_Helper::createAndRegisterWorkloadContext() {
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

inline void WorkloadContext_Helper::destroyHddlUniteContext(const WorkloadID &id) {
    HddlUnite::unregisterWorkloadContext(id);
}

inline WorkloadID WorkloadContext_Helper::getWorkloadId() const {
    return _workloadId;
}

inline bool WorkloadContext_Helper::isValidWorkloadContext(const WorkloadID &id) {
    HddlUnite::WorkloadContext::Ptr workloadContext = HddlUnite::queryWorkloadContext(id);
    return workloadContext != nullptr;
}

inline HddlUnite::WorkloadContext::Ptr WorkloadContext_Helper::getWorkloadContext() const {
    return HddlUnite::queryWorkloadContext(_workloadId);
}
