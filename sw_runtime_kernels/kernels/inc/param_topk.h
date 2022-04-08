#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct TopKParams {
    struct MemRefData inputValues;
    struct MemRefData k;
    struct MemRefData outputValues;
    struct MemRefData outputIndex;
    
    int64_t axis;
    int64_t mode;
    int64_t sort;
};

#pragma pack(pop)

inline struct BaseKernelParams ToBaseKernelParams(struct TopKParams* params) {
    struct BaseKernelParams result;
    result.numInputs = 1;
    result.numOutputs = 2;
#ifdef __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->inputValues)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(params->outputValues)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->inputValues)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->outputValues)) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
