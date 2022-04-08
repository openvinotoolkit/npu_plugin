//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <param_swish.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.
#define intrinsic_vec(intrinsic, vin, vout) \
    (vout)[0] = intrinsic((vin)[0]);        \
    (vout)[1] = intrinsic((vin)[1]);        \
    (vout)[2] = intrinsic((vin)[2]);        \
    (vout)[3] = intrinsic((vin)[3]);        \
    (vout)[4] = intrinsic((vin)[4]);        \
    (vout)[5] = intrinsic((vin)[5]);        \
    (vout)[6] = intrinsic((vin)[6]);        \
    (vout)[7] = intrinsic((vin)[7]);
#define exp2_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_exp2_f16_l_r, vin, vout)

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void swish_fp16(const struct SwishParams* lParams) {
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    const half p_act_beta = (half)(lParams->beta);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t nElements = 1;
    int32_t i = 0;

    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)&inv_ln2;

    const half8 one = (half8)1.0h;
    const half8 nbeta_inv_ln2 = (half8)(-p_act_beta * inv_ln2_h);

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    const int numVectors = nElements / VECTOR_SIZE;

    // Swish(x) = x / (1 + exp(-beta * x))
    for (int i = 0; i < numVectors; ++i) {
        const half8 x = p_act_data_v[i];

        half8 exponent = __builtin_shave_vau_mul_f16_rr(nbeta_inv_ln2, x);
        exp2_vec(exponent, exponent);

        half8 denom = __builtin_shave_vau_add_f16_rr(exponent, one);
        half8 inv_denom = 1 / denom;

        half8 res = __builtin_shave_vau_mul_f16_rr(x, inv_denom);

        p_act_out_v[i] = res;
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] =
                p_act_data_s[i] / (1.0f + __builtin_shave_sau_exp2_f16_l_r(-p_act_beta * p_act_data_s[i] * inv_ln2_h));
    }
}
}
}  // namespace shave_lib
}  // namespace nn
