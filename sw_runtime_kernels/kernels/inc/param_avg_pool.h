//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <moviVectorTypes.h>
#include <common_types.h>
#include <cstddef>

#ifdef __cplusplus
namespace sw_params {
#endif

#define MAX_ATTR_SZ 2 // max_dim(4) - 2

#pragma pack(push, 1)

struct AvgPoolParams {
    struct MemRefData input;
    struct MemRefData output;

    int64_t kernelSize[MAX_ATTR_SZ];
    int64_t strides   [MAX_ATTR_SZ];
    int64_t padsBegin [MAX_ATTR_SZ];
    int64_t padsEnd   [MAX_ATTR_SZ];
    int64_t exclude_pads;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct AvgPoolParams * params) {
    struct BaseKernelParams result;

    result.numInputs  = 1;
    result.numOutputs = 1;

    result.inputsOffset  = offsetof(AvgPoolParams, input);
    result.outputsOffset = offsetof(AvgPoolParams, output);

    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
