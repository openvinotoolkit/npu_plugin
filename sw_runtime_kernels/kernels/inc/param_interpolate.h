//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <common_types.h>
#include <mv_types.h>

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

enum InterpolationMethod : int32_t {
    NEAREST     = 0,
    BILINEAR    = 1,
    BICUBIC     = 2,
    LINEAR_ONNX = 3
};

enum InterpolationCoordTransMode : int32_t {
    HALF_PIXEL           = 0,
    PYTORCH_HALF_PIXEL   = 1,
    ASYMMETRIC           = 2,
    TF_HALF_PIXEL_FOR_NN = 3,
    ALIGN_CORNERS        = 4
};

enum InterpolationNearestMode : int32_t {
    ROUND_PREFER_FLOOR   = 0,
    ROUND_PREFER_CEIL    = 1,
    FLOOR                = 2,
    CEIL                 = 3,
    SIMPLE               = 4
};

namespace sw_params {

struct __attribute__((packed)) InterpolateParams {
    struct MemRefData input;
    struct MemRefData output;

    InterpolationMethod interpolation_mode;
    InterpolationCoordTransMode coord_transform_mode;
    InterpolationNearestMode nearest_mode;
    uint64_t antialias;
};

inline struct BaseKernelParams ToBaseKernelParams(struct InterpolateParams * params) {
    struct BaseKernelParams result;
    result.numInputs = 1;
    result.numOutputs = 1;
#ifdef  __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->input)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(params->output)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->input)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->output)) - (uint8_t*)(params);
#endif
    return result;
}

}
