//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorTypes.h>
#include <math.h>
#include <param_sau_dp4m.h>

using namespace sw_params;

extern "C"
void sau_dp4m(const struct SauDp4mParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + SauDp4mParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size

    schar4* in1 = (schar4*)(inputs[0].dataAddr);
    uchar4* in2 = (uchar4*)(inputs[1].dataAddr);
    int32_t* out = (int32_t*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        schar4 a = *in1++;
        uchar4 b = *in2++;
        *out++ = __builtin_shave_sau_dp4m_v4i8_rr(a, b);
    }
}
