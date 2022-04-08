//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorTypes.h>
#include <math.h>
#include <param_vau_exp.h>

using namespace sw_params;

extern "C"
void vau_exp_fp16(const struct VauExpParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + VauExpParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size
    nElements = nElements / 8;

    half8* in = (half8*)(inputs[0].dataAddr);
    half8* out = (half8*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        half8 a = *in++;
        *out++ = __builtin_shave_vau_exp_v8f16_r(a);
    }
}
