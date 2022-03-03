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

namespace sw_params {

struct __attribute__((packed)) ScatterNDUpdateParams {
    struct MemRefData input;
    struct MemRefData output;
};

inline struct BaseKernelParams ToBaseKernelParams(struct ScatterNDUpdateParams * params) {
    struct BaseKernelParams result;
    result.numInputs = 1; ///
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
