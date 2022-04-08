//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct SoftmaxParams {
    struct MemRefData input;
    struct MemRefData output;
    int64_t axis;
};

#pragma pack(pop)

inline BaseKernelParams softmaxParamsToBaseKernelParams(SoftmaxParams * softmaxParams) {
    BaseKernelParams rezult;
    rezult.numInputs = 1;
    rezult.numOutputs = 1;
    rezult.inputsOffset = reinterpret_cast<uint8_t*>(&(softmaxParams->input)) - reinterpret_cast<uint8_t*>(softmaxParams);
    rezult.outputsOffset = reinterpret_cast<uint8_t*>(&(softmaxParams->output)) - reinterpret_cast<uint8_t*>(softmaxParams);
    return rezult;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
