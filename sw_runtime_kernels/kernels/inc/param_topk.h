#pragma once

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>
#include <mv_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#define MAX_TK_DIMS 8

/// TopKMax parameters
struct __attribute__((packed)) TopKParams{
    struct MemRefData inputValues;
    struct MemRefData outputValues;
    struct MemRefData outputIndices;

    int32_t k;

    int32_t inNdims;
    int32_t outNdims;

    int32_t inputValueDims[MAX_TK_DIMS];
    int32_t outputValueDims[MAX_TK_DIMS];
    int32_t outputIndicesDims[MAX_TK_DIMS];

    int32_t inputValueStrides[MAX_TK_DIMS];
    int32_t outputValueStrides[MAX_TK_DIMS];
    int32_t outputIndicesStrides[MAX_TK_DIMS];

    int32_t inputDim;
    int32_t outputDim;
    int32_t inputValueStride;
    int32_t outputValueStride;
    int32_t outputIndicesStride;

    int32_t start;
    int32_t toProcess;

    int32_t mode;
    int32_t sort;
    int32_t axis;
    int32_t hasValues;
    int32_t hasIndices;
};

// TopK internal struct
typedef struct {
    int32_t index; // for n<2^16 u16 can be used with additional DMA zero filling of high index halves; it's slightly faster
    int32_t value;
} t_MvTopKPack;

inline BaseKernelParams TopKParamsToBaseKernelParams(TopKParams * topKParams) {
    BaseKernelParams results;
    results.numInputs = 1;
    results.numOutputs = 1;// change to 2 how? 
    results.inputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->input)) - reinterpret_cast<uint8_t*>(topKParams);
    results.outputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->output)) - reinterpret_cast<uint8_t*>(topKParams);
    return results;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif