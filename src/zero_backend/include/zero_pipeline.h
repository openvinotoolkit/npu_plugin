//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "zero_memory.h"
#include "zero_utils.h"

namespace vpux {
struct Pipeline {
public:
    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;
    virtual ~Pipeline() = default;

    virtual void push() = 0;
    virtual void pull() = 0;
    virtual void reset() const = 0;

    inline zeroMemory::MemoryManagementUnit& inputs() {
        return _inputs;
    };
    inline zeroMemory::MemoryManagementUnit& outputs() {
        return _outputs;
    };

protected:
    zeroMemory::MemoryManagementUnit _inputs;
    zeroMemory::MemoryManagementUnit _outputs;
};

}  // namespace vpux
