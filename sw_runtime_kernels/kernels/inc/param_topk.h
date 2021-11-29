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
enum TopKMode : int8_t {
    max = 0,
    min = 1
};

enum TopKSort : int8_t {
    none  = 0,
    value = 1,
    index = 2
};

/// TopKMax parameters
struct __attribute__((packed)) TopKParams{
    compat_ptr<const uint8_t> inputValues;
    compat_ptr<uint8_t> outputValues;
    compat_ptr<uint8_t> outputIndices;
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

    TopKMode mode;
    TopKSort sort;
    int32_t axis;
    bool hasValues;
    bool hasIndices;
};

// TopK internal struct
typedef struct {
    int32_t index; // for n<2^16 u16 can be used with additional DMA zero filling of high index halves; it's slightly faster
    fp16 value;
} t_MvTopKPack;

inline BaseKernelParams TopKParamsToBaseKernelParams(TopKParams * topKParams) {
    BaseKernelParams results;
    results.numInputs = 1;
    results.numOutputs = 1;
    results.inputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->input)) - reinterpret_cast<uint8_t*>(topKParams);
    results.outputsOffset = reinterpret_cast<uint8_t*>(&(topKParams->output)) - reinterpret_cast<uint8_t*>(topKParams);
    return results;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif