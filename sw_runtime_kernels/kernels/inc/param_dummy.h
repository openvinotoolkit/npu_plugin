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

#ifndef _DUMMY_PARAMS_
#define _DUMMY_PARAMS_

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

struct DummyParams {
    uint64_t numIns;
    uint64_t numOuts;
    struct MemRefData tensors[MAX_KERNEL_INPUTS + MAX_KERNEL_OUTPUTS];
};

#pragma pack (pop)

inline struct BaseKernelParams dummyParamsToBaseKernelParams(struct DummyParams * dummyParams) {
    struct BaseKernelParams rezult;
    rezult.numInputs = dummyParams->numIns;
    rezult.numOutputs = dummyParams->numOuts;
#ifdef  __cplusplus
    rezult.inputsOffset = reinterpret_cast<uint8_t*>(dummyParams->tensors) - reinterpret_cast<uint8_t*>(dummyParams);
    rezult.outputsOffset = reinterpret_cast<uint8_t*>(dummyParams->tensors + dummyParams->numIns) - reinterpret_cast<uint8_t*>(dummyParams);
#else
    rezult.inputsOffset = (uint8_t*)(dummyParams->tensors) - (uint8_t*)(dummyParams);
    rezult.outputsOffset = (uint8_t*)(dummyParams->tensors + dummyParams->numIns) - (uint8_t*)(dummyParams);
#endif
    return rezult;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif

#endif  // _DUMMY_PARAMS_
