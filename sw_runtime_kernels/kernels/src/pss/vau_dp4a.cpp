// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorTypes.h>
#include <math.h>
#include <param_vau_dp4a.h>

using namespace sw_params;

extern "C"
void vau_dp4a(const struct VauDp4aParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + VauDp4aParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size
    nElements = nElements / 4;

    schar16* in1 = (schar16*)(inputs[0].dataAddr);
    schar16* in2 = (schar16*)(inputs[1].dataAddr);
    int4* out = (int4*)(outputs[0].dataAddr);

    if (nElements > 0) {
        // zero the accumulator before MAC
        schar16 a = *in1++;
        schar16 b = *in2++;
        *out++ = __builtin_shave_vau_dp4az_v16i8_vacc0_rr(a, b);
    }
    for (int32_t e = 1; e < nElements; ++e) {
        // retain previous accumulator value
        schar16 a = *in1++;
        schar16 b = *in2++;
        *out++ = __builtin_shave_vau_dp4a_v16i8_vacc0_rr(a, b);
    }
}
