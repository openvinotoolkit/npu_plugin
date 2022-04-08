//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_hardsigmoid.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace sw_params;
namespace nn {
namespace shave_lib {

extern "C" {

void hardsigmoid_fp16(uint32_t lParamsAddr) {
    const HardSigmoidParams* lParams = reinterpret_cast<const HardSigmoidParams*>(lParamsAddr);

    half8* p_act_data8 = (half8*)(lParams->input.dataAddr);
    half8* p_act_out8 = (half8*)(lParams->output.dataAddr);

    half* p_act_data = (half*)(lParams->input.dataAddr);
    half* p_act_out = (half*)(lParams->output.dataAddr);

    const half alpha = lParams->alpha;
    const half beta = lParams->beta;
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    const half8 one = (half8)1.0f;
    const half8 zero = (half8)0.0f;
    int32_t nElements = 1;
    int32_t i = 0;
    half act = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    const int numVectors = nElements / VECTOR_SIZE;

#pragma clang loop unroll_count(8)
    for (i = 0; i < numVectors; i++) {
        half8 res8 = p_act_data8[i] * alpha + beta;
        res8 = __builtin_shave_cmu_min_f16_rr_half8(res8, one);
        p_act_out8[i] = __builtin_shave_cmu_max_f16_rr_half8(res8, zero);
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        half res =  p_act_data[i] * alpha + beta;
        res = MIN(1.0f, res);
        p_act_out[i] = MAX(0.0f, res);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
