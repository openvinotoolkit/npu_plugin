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

#include "vpux/utils/IE/loop.hpp"

#include <ie_parallel.hpp>

using namespace vpux;
using namespace InferenceEngine;

StringLiteral vpux::stringifyEnum(LoopExecPolicy val) {
#define CASE(_val_)             \
    case LoopExecPolicy::_val_: \
        return #_val_

    switch (val) {
        CASE(Sequential);
        CASE(Parallel);
    default:
        return "<UNKNOWN>";
    }

#undef CASE
}

void vpux::loop_1d(LoopExecPolicy policy, int64_t dim0, FuncRef<void(int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        parallel_for(dim0, proc);
    } else {
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            proc(d0);
        }
    }
}

void vpux::loop_2d(LoopExecPolicy policy, int64_t dim0, int64_t dim1, FuncRef<void(int64_t, int64_t)> proc) {
    if (policy == LoopExecPolicy::Parallel) {
        parallel_for2d(dim0, dim1, proc);
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
        parallel_for3d(dim0, dim1, dim2, proc);
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
        parallel_for4d(dim0, dim1, dim2, dim3, proc);
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
