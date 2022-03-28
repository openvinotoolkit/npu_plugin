//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorTypes.h>
#include <math.h>
#include <param_sigmoid.h>

void sigmoid_fp16(const struct SigmoidParams *lParams) {
//    uint32_t * tmp = (uint32_t *)0x2e014000;
//    uint32_t * debInd = tmp;
//    uint32_t * dims = (uint32_t*)(lParams->input.dimsAddr);
//
//    *debInd = 1;
//
//    if (tmp[*debInd] == 555555) {
//        (*debInd)++;
//        tmp[(*debInd)++]++;
//    } else {
//        tmp[(*debInd)++] = 555555;
//        tmp[(*debInd)++] = 0;
//    }
//    tmp[(*(debInd))++] = (uint32_t)(lParams->input.dimsAddr);
//    tmp[(*(debInd))++] = (uint32_t)(dims[0]);
//    tmp[(*(debInd))++] = (uint32_t)(dims[1]);
//    tmp[(*(debInd))++] = (uint32_t)(dims[2]);
//    tmp[(*(debInd))++] = (uint32_t)(dims[3]);
//    return;

    half* p_act_data = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;
    half act = 0;

    for (i = 0; i!= lParams->input.numDims; i++ ) {
        // TODO: check overflow
        nElements *=  pDims[i];
    }

    for (uint32_t e = 0; e < nElements; ++e) {
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        *p_act_out++ = (half)act;
    }
}
