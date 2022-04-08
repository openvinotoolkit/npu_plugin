//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorConvert.h>
#include <param_trigonometric.h>

using namespace sw_params;

inline half trigonometric_scl_fp16(half a);

void trigonometric_fp16(const struct TrigonometricParams* lParams) {
    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    for (i = 0; i < nElements; i++) {
        p_act_out_s[i] = trigonometric_scl_fp16(p_act_data_s[i]);
    }
}
