//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

enum class LoopExecPolicy {
    Sequential,
    Parallel,
};

StringLiteral stringifyEnum(LoopExecPolicy val);

void loop_1d(LoopExecPolicy policy, int64_t dim0, FuncRef<void(int64_t)> proc);

void loop_2d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, FuncRef<void(int64_t, int64_t)> proc);

void loop_3d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2,
             FuncRef<void(int64_t, int64_t, int64_t)> proc);

void loop_4d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
             FuncRef<void(int64_t, int64_t, int64_t, int64_t)> proc);

}  // namespace vpux
