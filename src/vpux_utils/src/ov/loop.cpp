//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/loop.hpp"

#include <openvino/core/parallel.hpp>

using namespace vpux;

StringLiteral vpux::stringifyEnum(LoopExecPolicy val) {
    switch (val) {
    case LoopExecPolicy::Sequential:
        return "Sequential";
    case LoopExecPolicy::Parallel:
        return "Parallel";
    default:
        return "<UNKNOWN>";
    }
}

void vpux::loop_1d(LoopExecPolicy policy, int64_t dim0, FuncRef<void(int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        ov::parallel_for(dim0, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            proc(d0);
        }
    }
}

void vpux::loop_2d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, FuncRef<void(int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        ov::parallel_for2d(dim0, dim1, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                proc(d0, d1);
            }
        }
    }
}
void vpux::loop_3d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2,
                   FuncRef<void(int64_t, int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        ov::parallel_for3d(dim0, dim1, dim2, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                for (int64_t d2 = 0; d2 < dim2; ++d2) {
                    proc(d0, d1, d2);
                }
            }
        }
    }
}

void vpux::loop_4d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                   FuncRef<void(int64_t, int64_t, int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        ov::parallel_for4d(dim0, dim1, dim2, dim3, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            for (int64_t d1 = 0; d1 < dim1; ++d1) {
                for (int64_t d2 = 0; d2 < dim2; ++d2) {
                    for (int64_t d3 = 0; d3 < dim3; ++d3) {
                        proc(d0, d1, d2, d3);
                    }
                }
            }
        }
    }
}
