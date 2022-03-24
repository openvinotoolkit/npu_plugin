// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorTypes.h>
#include <math.h>
#include <param_sau_dp4a.h>

using namespace sw_params;

extern "C"
void sau_dp4a(const struct SauDp4aParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + SauDp4aParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size

    schar4* in1 = (schar4*)(inputs[0].dataAddr);
    schar4* in2 = (schar4*)(inputs[1].dataAddr);
    int32_t* out = (int32_t*)(outputs[0].dataAddr);

    if (nElements > 0) {
        // zero the accumulator before MAC
        schar4 a = *in1++;
        schar4 b = *in2++;
        *out++ = __builtin_shave_sau_dp4az_v4i8_sacc0_rr(a, b);
    }
    for (int32_t e = 1; e < nElements; ++e) {
        // retain previous accumulator value
        schar4 a = *in1++;
        schar4 b = *in2++;
        *out++ = __builtin_shave_sau_dp4a_v4i8_sacc0_rr(a, b);
    }
}
