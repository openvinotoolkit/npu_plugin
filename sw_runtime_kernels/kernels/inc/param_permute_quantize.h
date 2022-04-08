//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

enum PermuteQuantizeOptMode : int64_t {
    PQ_NONE = 0,
    PQ_NCHW_NHWC_C1 = 1,
    PQ_NCHW_NHWC_C3 = 2,
    PQ_NCHW_NHWC_C4 = 3,
    PQ_NCHW_NHWC_C3EXP4 = 4,
    PQ_NCHW_NHWC_C4EXP4 = 5,
    PQ_NCHW_NHWC_C1EXP4 = 6
};

#pragma pack(push, 1)
struct PermuteQuantizeParams {
    struct MemRefData input;
    struct MemRefData output;

    int64_t opt_mode;
    float scale;
    int64_t zero;
    int64_t perm[MAX_ND_DIMS];
};
#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct PermuteQuantizeParams* params) {
    struct BaseKernelParams result;
    result.numInputs = 1;
    result.numOutputs = 1;
#ifdef __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->input)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(params->output)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->input)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->output)) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
