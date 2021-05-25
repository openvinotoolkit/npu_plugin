//
// Copyright 2020 Intel Corporation.
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
