//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

struct __attribute__((packed)) LRNParams {
    struct MemRefData input;
    struct MemRefData axis;
    struct MemRefData output;

    float alpha;
    float beta;
    float bias;
    int64_t size;
};

inline struct BaseKernelParams ToBaseKernelParams(struct LRNParams* params) {
    struct BaseKernelParams result;
    result.numInputs = 2;
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