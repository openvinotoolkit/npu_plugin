//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"

namespace vpux {
class VariableState : public InferenceEngine::IVariableStateInternal {
public:
    VariableState(const std::string& name, InferenceEngine::Blob::Ptr state)
            : InferenceEngine::IVariableStateInternal{name} {
        this->state = state;
    }

    void Reset() override;
    void SetState(const InferenceEngine::Blob::Ptr& new_state) override;
    InferenceEngine::Blob::CPtr GetState() const override;

    virtual ~VariableState() = default;
};
}  // namespace vpux
