//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>
#include <moviVectorTypes.h>
#include <cstddef>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

#define MAX_ATTR_SZ 2  // max_dim(4) - 2

struct __attribute__((packed)) MaxPoolParams {
    struct MemRefData input;
    struct MemRefData output;

    int64_t kernelSize[MAX_ATTR_SZ];
    int64_t strides[MAX_ATTR_SZ];
    int64_t padsBegin[MAX_ATTR_SZ];
    int64_t padsEnd[MAX_ATTR_SZ];
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct MaxPoolParams* params) {
    struct BaseKernelParams result;

    result.numInputs = 1;
    result.numOutputs = 1;

    result.inputsOffset = offsetof(MaxPoolParams, input);
    result.outputsOffset = offsetof(MaxPoolParams, output);

    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
