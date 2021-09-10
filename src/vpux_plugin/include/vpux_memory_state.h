// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blob_factory.hpp"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"

#include <string>

namespace vpux {

class VPUXVariableState : public InferenceEngine::IVariableStateInternal {
public:
    VPUXVariableState(const std::string& name, const InferenceEngine::Blob::Ptr& blob)
            : InferenceEngine::IVariableStateInternal{name} {
        state = blob;
    }

    void SetState(const InferenceEngine::Blob::Ptr& newState) override;
    void Reset() override;

    virtual ~VPUXVariableState() = default;
};

}  // namespace vpux
