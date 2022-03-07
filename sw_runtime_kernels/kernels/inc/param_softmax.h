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

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

struct __attribute__((packed)) SoftmaxParams {
    struct MemRefData input;
    struct MemRefData output;
    int64_t axis;
};

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
