//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>
#include <cstddef>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct ReduceParams {
    struct MemRefData input;
    struct MemRefData axes;
    struct MemRefData output;
    int64_t keep_dims;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct ReduceParams * params) {
    (void)*params;
    struct BaseKernelParams result;
    result.numInputs  = 2;
    result.numOutputs = 1;

    result.inputsOffset  = offsetof(ReduceParams, input);
    result.outputsOffset = offsetof(ReduceParams, output);

    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
