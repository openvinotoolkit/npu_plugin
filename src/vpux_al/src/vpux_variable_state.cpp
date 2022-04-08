//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <cstdint>

#include <ie_blob.h>

#include <vpux_variable_state.hpp>

namespace vpux {
void VariableState::Reset() {
    IE_ASSERT(state != nullptr);

    auto mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(state);
    IE_ASSERT(mem_blob);
    auto state_mem_locker = mem_blob->wmap();
    std::uint8_t* state_buffer = state_mem_locker.as<std::uint8_t*>();

    std::memset(state_buffer, 0, state->byteSize());
}

void VariableState::SetState(const InferenceEngine::Blob::Ptr& new_state) {
    IE_ASSERT(new_state != nullptr);
    IE_ASSERT(state != nullptr);

    auto new_state_mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(new_state);
    IE_ASSERT(new_state_mem_blob);
    auto new_state_mem_locker = new_state_mem_blob->rmap();
    const std::uint8_t* new_state_buffer = new_state_mem_locker.as<const std::uint8_t*>();

    auto state_mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(state);
    IE_ASSERT(state_mem_blob);
    auto state_mem_locker = state_mem_blob->wmap();
    std::uint8_t* state_buffer = new_state_mem_locker.as<std::uint8_t*>();

    IE_ASSERT(new_state->byteSize() == state->byteSize());
    std::memcpy(state_buffer, new_state_buffer, state->byteSize());
}

InferenceEngine::Blob::CPtr VariableState::GetState() const {
    return state;
}
}  // namespace vpux
