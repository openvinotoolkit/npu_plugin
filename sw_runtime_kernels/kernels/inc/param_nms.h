//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct NMSParams {
    struct MemRefData boxes;
    struct MemRefData scores;
    struct MemRefData selectedIndices;
    struct MemRefData selectedScores;
    struct MemRefData validOutputs;
    int64_t maxOutputBoxesPerClass;
    float iouThreshold;
    float scoreThreshold;
    float softNmsSigma;
    int64_t boxEncoding;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct NMSParams* params) {
    struct BaseKernelParams result;
    result.numInputs = 2;
    result.numOutputs = 3;
#ifdef __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->boxes)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(params->selectedIndices)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->boxes)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->selectedIndices)) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
