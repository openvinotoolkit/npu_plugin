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

#include <nn_log.h>

#ifndef VECTOR_SIZE
#define VECTOR_SIZE    8
#endif

#if defined(ELTWISE_VEC_OP)
#define VECTOR_LOOP                                       \
 {                                                        \
    half8* vecInA = (half8*)(p->input[0].dataAddr);       \
    half8* vecInB = (half8*)(p->input[1].dataAddr);       \
    half8* vecOut = (half8*)(p->output.dataAddr);         \
    const uint32_t numVectors = nElements / VECTOR_SIZE;  \
                                                          \
    _Pragma("clang loop unroll_count(8)")                 \
    for (i = 0; i < numVectors; i++) {                    \
        vecOut[i] = ELTWISE_VEC_OP(vecInA[i], vecInB[i]); \
    }                                                     \
    i = i * VECTOR_SIZE;                                  \
  }
#else
#define VECTOR_LOOP //nothing, just run scalar-loop
#endif



#define ELTWISE_BINARY_OP(funcName)                     \
                                                        \
void funcName(const struct EltwiseParams *p) {          \
                                                        \
    half* inA = (half*)(p->input[0].dataAddr);          \
    half* inB = (half*)(p->input[1].dataAddr);          \
    half* out = (half*)(p->output.dataAddr);            \
                                                        \
    int32_t *pDims = (int32_t *)(p->input[0].dimsAddr); \
    uint32_t nElements = 1;                             \
                                                        \
    for (uint32_t i=0; i < p->input[0].numDims; i++) {  \
        nElements *= pDims[i];                          \
    }                                                   \
                                                        \
    uint32_t i = 0;                                     \
    VECTOR_LOOP                                         \
                                                        \
    for (; i<nElements; i++) {                          \
        out[i] = ELTWISE_SCL_OP(inA[i], inB[i]);        \
    }                                                   \
}
