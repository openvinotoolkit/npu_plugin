#pragma once

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct TopKParams {
    struct MemRefData input;
    struct MemRefData k;
    struct MemRefData value;
    struct MemRefData index;
    
    int32_t mode;
    int32_t sort;
    int32_t axis;
    int32_t hasValues;
    int32_t hasIndices;
};

#pragma pack (pop)

inline struct BaseKernelParams ToBaseKernelParams(struct TopKParams * params) {
    struct BaseKernelParams result;
    result.numInputs = 2;
    result.numOutputs = 2;
#ifdef  __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->input)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(params->value)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->input)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->value)) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif