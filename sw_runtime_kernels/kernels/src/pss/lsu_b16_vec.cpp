// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorTypes.h>
#include <math.h>
#include <param_lsu_b16_vec.h>

using namespace sw_params;

extern "C"
void lsu_b16_vec(const struct LsuB16VecParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + LsuB16VecParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size
    nElements = nElements / 4;

    half4* in = (half4*)(inputs[0].dataAddr);
    half4* out = (half4*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        float4 val = __builtin_shave_lsu_ld128_b16_f32_r(in++);
        __builtin_shave_lsu_st64_f32_b16_rr(val, out++);
    }
}
