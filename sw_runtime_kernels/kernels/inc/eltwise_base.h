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

#define ELTWISE_MATH_2_OP(funcName)                           \
                                                              \
void funcName(const struct EltwiseParams *lParams) {          \
                                                              \
    half* inA = (half*)(lParams->input[0].dataAddr);          \
    half* inB = (half*)(lParams->input[1].dataAddr);          \
    half* out = (half*)(lParams->output.dataAddr);            \
                                                              \
    int32_t *pDims = (int32_t *)(lParams->input[0].dimsAddr); \
    uint32_t nElements = 1;                                   \
                                                              \
    for (uint32_t i=0; i < lParams->input[0].numDims; i++) {  \
        nElements *= pDims[i];                                \
    }                                                         \
                                                              \
    for (uint32_t i=0; i < nElements; i++) {                  \
        out[i] = ELTWISE_FN(inA[i],inB[i]);                   \
    }                                                         \
}
