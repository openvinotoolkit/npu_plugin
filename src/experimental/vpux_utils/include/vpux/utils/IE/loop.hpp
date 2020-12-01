//
// Copyright 2020 Intel Corporation.
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
