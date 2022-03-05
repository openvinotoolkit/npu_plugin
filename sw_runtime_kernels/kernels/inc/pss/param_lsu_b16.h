//
// Copyright Intel Corporation.
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

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct LsuB16Params {
    enum { NumInputs = 1, NumOutputs = 1, NumTensors = NumInputs + NumOutputs };
    struct MemRefData tensors[NumTensors];
};

#pragma pack (pop)

inline struct BaseKernelParams ToBaseKernelParams(struct LsuB16Params * params) {
    struct BaseKernelParams result;
    result.numInputs = LsuB16Params::NumInputs;
    result.numOutputs = LsuB16Params::NumOutputs;
#ifdef  __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(params->tensors + 0) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(params->tensors + LsuB16Params::NumInputs) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(params->tensors + 0) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(params->tensors + LsuB16Params::NumInputs) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
