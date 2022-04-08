//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_trigonometric.h>

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif

#define intrinsic_vau_vec(intrinsic, vin, vout) (vout) = intrinsic((vin))
#define exp_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_exp_v8f16_r, vin, vout))

using namespace sw_params;
namespace nn {
namespace shave_lib {

extern "C" {

void sinh_fp16(uint32_t lParamsAddr) {
    const TrigonometricParams* lParams = (const TrigonometricParams*)lParamsAddr;
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    const int numVectors = nElements / VECTOR_SIZE;
    const half8 divBy2 = (half8)0.5f;

    for (i = 0; i < numVectors; i++) {
        half8 vin = p_act_data_v[i];
        half8 exp_positive;
        half8 exp_negative;
        half8 vout;
        // (e^x - e^(-x))/2
        exp_vec(vin, exp_positive);
        exp_vec((-vin), exp_negative);
        vout = __builtin_shave_vau_sub_f16_rr(exp_positive, exp_negative);
        p_act_out_v[i] = vout * divBy2;
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        half in = p_act_data_s[i];
        half exp_positive_s;
        half exp_negative_s;
        half out;
        // (e^x - e^(-x))/2
        exp_positive_s = __builtin_shave_sau_exp_f16_r(in);
        exp_negative_s = __builtin_shave_sau_exp_f16_r(-in);

        out = exp_positive_s - exp_negative_s;
        p_act_out_s[i] = out * 0.5f;
    }
}
}
}  // namespace shave_lib
}  // namespace nn
