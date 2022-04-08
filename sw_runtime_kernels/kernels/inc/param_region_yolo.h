//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#define MAX_MASK_SIZE 9
#define MAX_ANCHOR_SIZE 18  // max prior box sizes * Size[width, height] = 9 * 2;

#pragma pack(push, 1)

struct RegionYoloParams {
    struct MemRefData input;
    struct MemRefData output;

    int64_t coords;
    int64_t classes;
    int64_t regions;
    uint64_t do_softmax;
    int64_t mask_size;
    int64_t mask[MAX_MASK_SIZE];
    int64_t axis;
    int64_t end_axis;
    float anchors[MAX_ANCHOR_SIZE];
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct RegionYoloParams* params) {
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
