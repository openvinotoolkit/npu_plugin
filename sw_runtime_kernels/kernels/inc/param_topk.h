#pragma once

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>
#include <mv_types.h>
#define SHAVE_LIB_DATA_SIZE 112 * 1024
#ifdef __cplusplus
namespace sw_params {
#endif

#define MAX_TK_DIMS 8

/// TopKMax parameters

struct __attribute__((packed)) TopKParams {
    struct MemRefData inputValues;
    struct MemRefData k;
    struct MemRefData outputValues;
    struct MemRefData outputIndices;
    
    int32_t start;
    int32_t toProcess;
    
    int32_t mode;
    int32_t sort;
    int32_t axis;
    int32_t hasValues;
    int32_t hasIndices;
};

inline BaseKernelParams TopKParamsToBaseKernelParams(TopKParams * topKParams) {
    BaseKernelParams result;
    result.numInputs = 2;
    result.numOutputs = 2;// change to 2 how? 
#ifdef  __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->inputValues)) - reinterpret_cast<uint8_t*>(topKParams);
    result.outputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->outputValues)) - reinterpret_cast<uint8_t*>(topKParams);
#else
    result.inputsOffset = (uint8_t*)(&(topKParams->inputValues)) - (uint8_t*)(topKParams);
    result.outputsOffset = (uint8_t*)(&(topKParams->outputValues)) - (uint8_t*)(topKParams);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif