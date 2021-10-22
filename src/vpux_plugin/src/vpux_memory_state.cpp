// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux_memory_state.h"
#include "blob_factory.hpp"

using namespace InferenceEngine;

namespace vpux {

void VPUXVariableState::SetState(const InferenceEngine::Blob::Ptr& newState) {
    ie_memcpy(state->buffer(), state->byteSize(), newState->buffer(), newState->byteSize());
}

void VPUXVariableState::WriteToState(InferenceEngine::Blob::Ptr& dstState) {
    // ie_memcpy(state->buffer(), state->byteSize(), newState->buffer(), newState->byteSize());
    ie_memcpy(dstState->buffer(), dstState->byteSize(), state->buffer(), state->byteSize());
}


void VPUXVariableState::Reset() {
    std::memset(state->buffer(), 0, state->byteSize());
}

}  // namespace vpux
