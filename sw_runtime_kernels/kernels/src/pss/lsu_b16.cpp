// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorTypes.h>
#include <math.h>
#include <param_lsu_b16.h>

using namespace sw_params;

extern "C"
void lsu_b16(const struct LsuB16Params *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + LsuB16Params::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    half* in = (half*)(inputs[0].dataAddr);
    half* out = (half*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        float val = __builtin_shave_lsu_ld32_b16_f32_r(in++);
        __builtin_shave_lsu_st16_f32_b16_rr(val, out++);
    }
}
