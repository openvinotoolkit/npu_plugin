//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_tanh.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define intrinsic_vau_vec(intrinsic, vin, vout) \
    (vout) = intrinsic((vin))
#define intrinsic_sau_vec(intrinsic, vin, vout) \
    (vout)[0] = intrinsic((vin)[0]);        \
    (vout)[1] = intrinsic((vin)[1]);        \
    (vout)[2] = intrinsic((vin)[2]);        \
    (vout)[3] = intrinsic((vin)[3]);        \
    (vout)[4] = intrinsic((vin)[4]);        \
    (vout)[5] = intrinsic((vin)[5]);        \
    (vout)[6] = intrinsic((vin)[6]);        \
    (vout)[7] = intrinsic((vin)[7]);

#ifdef USE_3720_INTSTRUCTIONS
#define tanh_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_tanh_v8f16_r, vin, vout))
#else
#define tanh_vec(vin, vout) {\
    const half8 upper_bound_v =   5.5f;\
    const half8 lower_bound_v = -10.5f;\
    const uint16_t inv_ln2 = 0x41c5;\
    const half inv_ln2_h = *(const half*)&inv_ln2;\
    const half one = (half)1.0f;\
    vin = __builtin_shave_cmu_clampab_f16_rrr_half8(vin, lower_bound_v, upper_bound_v);\
    vin = vin * inv_ln2_h;\
    intrinsic_sau_vec(__builtin_shave_sau_exp2_f16_l_r, vin, vin);\
    vout = (vin - one) / (vin  + one);\
}
#endif

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void tanh_fp16(const struct TanhParams *lParams) {

    half8* p_act_data_v = (half8*)(lParams->input.dataAddr); // 0x1F000000
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr); // 0x1F004000

    half* p_act_data_s = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out_s = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i!= lParams->input.numDims; i ++ ) {
        // TODO: check overflow
        nElements *=  pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;

    // Clamp the input to avoid fp16 precision overflow when computing exp.
    // This should not affect the results
    const half upper_bound_s =   5.5f;
    const half lower_bound_s = -10.5f;

    // Compute tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // = (2^(2x/ln(2)) - 1) / (2^(2x/ln(2)) + 1)
    const uint16_t inv_ln2_mul_2 = 0x41c5;
    const half inv_ln2_mul_2_h = *(const half*)&inv_ln2_mul_2;
    const half one = (half)1.0f;

#pragma clang loop unroll_count(8)
    for (i = 0; i < numVectors; i ++) {
        half8 vin = p_act_data_v[i];
        half8 vout;
        tanh_vec(vin, vout);
        p_act_out_v[i] = vout;
    }

    for(i = numVectors * VECTOR_SIZE; i < nElements; i ++){
        half result = MIN(upper_bound_s, MAX(lower_bound_s, p_act_data_s[i]));
        result = result * inv_ln2_mul_2_h;
        result = __builtin_shave_sau_exp2_f16_l_r(result);
        p_act_out_s[i] = (result - one) / (result + one);
    }

}

}
}  // namespace shave_lib
}  // namespace nn
