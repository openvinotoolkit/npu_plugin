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

struct EltwiseParams {
    struct MemRefData input[2];
    struct MemRefData output;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct EltwiseParams * params) {
    (void)*params;
    struct BaseKernelParams result;
    result.numInputs  = 2;
    result.numOutputs = 1;

    result.inputsOffset  = offsetof(EltwiseParams, input);
    result.outputsOffset = offsetof(EltwiseParams, output);

    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
