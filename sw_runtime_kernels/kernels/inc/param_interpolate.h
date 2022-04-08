//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#include <common_types.h>
#include <mv_types.h>

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

enum InterpolationMethod : int64_t { NEAREST = 0, BILINEAR = 1, LINEAR_ONNX = 2, BICUBIC = 3 };

enum InterpolationCoordTransMode : int64_t {
    HALF_PIXEL = 0,
    PYTORCH_HALF_PIXEL = 1,
    ASYMMETRIC = 2,
    TF_HALF_PIXEL_FOR_NN = 3,
    ALIGN_CORNERS = 4
};

enum InterpolationNearestMode : int64_t {
    ROUND_PREFER_FLOOR = 0,
    ROUND_PREFER_CEIL = 1,
    FLOOR = 2,
    CEIL = 3,
    SIMPLE = 4
};

namespace sw_params {

#pragma pack(push, 1)

struct InterpolateParams {
    struct MemRefData input;
    struct MemRefData output;

    InterpolationMethod interpolation_mode;
    InterpolationCoordTransMode coord_transform_mode;
    InterpolationNearestMode nearest_mode;
    int64_t antialias;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct InterpolateParams* params) {
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

}  // namespace sw_params
