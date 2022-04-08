//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// template limitations:
// it expects both inputs to be 4D and broadcastable
//

#include <param_eltwise.h>
#include <moviVectorConvert.h>

using namespace sw_params;

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif
#define ELTWISE_MAX_ND_DIMS 4

inline half eltwise_scl_fp16(half a, half b);

#if defined(ELTWISE_VEC_OP)
inline half8 eltwise_vec_fp16(half8 a, half8 b);
#endif

void eltwise_line_vec_to_vec_fp16(half* input1, half* input2, half* output, int size) {
    int i = 0;

#if defined(ELTWISE_VEC_OP)
    half8* inputVec1 = reinterpret_cast<half8*>(input1);
    half8* inputVec2 = reinterpret_cast<half8*>(input2);
    half8* outputVec = reinterpret_cast<half8*>(output);

    int numVectors = size / VECTOR_SIZE;
    for(; i < numVectors; i++){
        outputVec[i] = eltwise_vec_fp16(inputVec1[i], inputVec2[i]);
    }
    i *= VECTOR_SIZE;
#endif

    for(; i < size; i++) {
        output[i] = eltwise_scl_fp16(input1[i], input2[i]);
    }
}

void eltwise_line_vec_to_scl_fp16(half* input1, half* input2, half* output, int size) {
    int i = 0;

#if defined(ELTWISE_VEC_OP)
    half8* inputVec1 = reinterpret_cast<half8*>(input1);
    half8 inputVec2 = mvuConvert_half((float) *input2);
    half8* outputVec = reinterpret_cast<half8*>(output);

    int numVectors = size / VECTOR_SIZE;
    for(; i < numVectors; i++){
        outputVec[i] = eltwise_vec_fp16(inputVec1[i], inputVec2);
    }
    i *= VECTOR_SIZE;
#endif

    for(; i < size; i++) {
        output[i] = eltwise_scl_fp16(input1[i], *input2);
    }
}

void eltwise_line_scl_to_vec_fp16(half* input1, half* input2, half* output, int size) {
    int i = 0;

#if defined(ELTWISE_VEC_OP)
    half8 inputVec1 = mvuConvert_half((float) *input1);
    half8* inputVec2 = reinterpret_cast<half8*>(input2);
    half8* outputVec = reinterpret_cast<half8*>(output);

    int numVectors = size / VECTOR_SIZE;
    for(; i < numVectors; i++){
        outputVec[i] = eltwise_vec_fp16(inputVec1, inputVec2[i]);
    }
    i *= VECTOR_SIZE;
#endif

    for(; i < size; i++) {
        output[i] = eltwise_scl_fp16(*input1, input2[i]);
    }
}

void convertNumBitsToNumElements(int64_t* inStrides, int64_t* outStrides, uint32_t n) {
    for(int i = 0; i < n; i++)
        outStrides[i] = inStrides[i] / (8 * sizeof(half));
}

void eltwise_fp16(const struct EltwiseParams *lParams) {

    half* inputData1 = reinterpret_cast<half*>(lParams->input[0].dataAddr);
    half* inputData2 = reinterpret_cast<half*>(lParams->input[1].dataAddr);
    half* outputData = reinterpret_cast<half*>(lParams->output.dataAddr);
    int32_t* inputDims1 = (int32_t*)(lParams->input[0].dimsAddr);
    int32_t* inputDims2 = (int32_t*)(lParams->input[1].dimsAddr);
    int32_t* outputDims = (int32_t*)(lParams->output.dimsAddr);
    uint32_t inputNumDims1 = lParams->input[0].numDims;
    uint32_t inputNumDims2 = lParams->input[1].numDims;
    uint32_t outputNumDims = lParams->output.numDims;
    int64_t* inputStridesAddr1 = (int64_t*)(lParams->input[0].stridesAddr);
    int64_t* inputStridesAddr2 = (int64_t*)(lParams->input[1].stridesAddr);
    int64_t* outputStridesAddr = (int64_t*)(lParams->output.stridesAddr);

    int64_t inputStrides1[ELTWISE_MAX_ND_DIMS];
    int64_t inputStrides2[ELTWISE_MAX_ND_DIMS];
    int64_t outputStrides[ELTWISE_MAX_ND_DIMS];
    convertNumBitsToNumElements(inputStridesAddr1, inputStrides1, inputNumDims1);
    convertNumBitsToNumElements(inputStridesAddr2, inputStrides2, inputNumDims2);
    convertNumBitsToNumElements(outputStridesAddr, outputStrides, outputNumDims);

    int inBatchOffset1, inBatchOffset2;
    int inPlaneOffset1, inPlaneOffset2;
    int inLineOffset1, inLineOffset2;
    int inElemOffset1, inElemOffset2;

    int inBatchStep1 = inputDims1[3] == 1 ? 0 : 1;
    int inBatchStep2 = inputDims2[3] == 1 ? 0 : 1;
    int inPlaneStep1 = inputDims1[2] == 1 ? 0 : 1;
    int inPlaneStep2 = inputDims2[2] == 1 ? 0 : 1;
    int inLineStep1 = inputDims1[1] == 1 ? 0 : 1;
    int inLineStep2 = inputDims2[1] == 1 ? 0 : 1;
    int inElemStep1 = inputDims1[0] == 1 ? 0 : 1;
    int inElemStep2 = inputDims2[0] == 1 ? 0 : 1;

    int outOffset = 0;
    inBatchOffset1 = inBatchOffset2 = 0;

    for(int batch = 0; batch < outputDims[3]; batch++){
        inPlaneOffset1 = inPlaneOffset2 = 0;
        for(int planes = 0; planes < outputDims[2]; planes++){
            inLineOffset1 = inLineOffset2 = 0;
            for(int lines = 0; lines < outputDims[1]; lines++){
                int inOffset1 = inBatchOffset1 + inPlaneOffset1 + inLineOffset1;
                int inOffset2 = inBatchOffset2 + inPlaneOffset2 + inLineOffset2;
                int lineSize = outputDims[0];

                // This branch condition can be moved outside of loop and use a pointer to function here
                // But for some unknown reason, on movisim, the shave hangs if I try to use pointers to
                // functions combined with branching
                // C++ alternatives like std::function are too heavy to import?
                if(inElemStep1 == 0) {
                    eltwise_line_scl_to_vec_fp16(&inputData1[inOffset1], &inputData2[inOffset2], &outputData[outOffset], lineSize);
                } else if(inElemStep2 == 0) {
                    eltwise_line_vec_to_scl_fp16(&inputData1[inOffset1], &inputData2[inOffset2], &outputData[outOffset], lineSize);
                } else {
                    eltwise_line_vec_to_vec_fp16(&inputData1[inOffset1], &inputData2[inOffset2], &outputData[outOffset], lineSize);
                }

                outOffset += lineSize;
                inLineOffset1 += inLineStep1 * inputStrides1[1];
                inLineOffset2 += inLineStep2 * inputStrides2[1];
            }
            inPlaneOffset1 += inPlaneStep1 * inputStrides1[2];
            inPlaneOffset2 += inPlaneStep2 * inputStrides2[2];
        }
        inBatchOffset1 += inBatchStep1 * inputStrides1[3];
        inBatchOffset2 += inBatchStep2 * inputStrides2[3];
    }
}
