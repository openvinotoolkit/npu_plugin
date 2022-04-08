//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_log.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

#define intrinsic_vau_vec(intrinsic, vin, vout) (vout) = intrinsic((vin))
#define intrinsic_sau_vec(intrinsic, vin, vout) \
    (vout)[0] = intrinsic((vin)[0]);            \
    (vout)[1] = intrinsic((vin)[1]);            \
    (vout)[2] = intrinsic((vin)[2]);            \
    (vout)[3] = intrinsic((vin)[3]);            \
    (vout)[4] = intrinsic((vin)[4]);            \
    (vout)[5] = intrinsic((vin)[5]);            \
    (vout)[6] = intrinsic((vin)[6]);            \
    (vout)[7] = intrinsic((vin)[7]);

#ifdef USE_3720_INTSTRUCTIONS
#define log_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_log_v8f16_r, vin, vout))
#else
#define log_vec(vin, vout) \
    { intrinsic_sau_vec(__builtin_shave_sau_log_f16_r, vin, vout); }
#endif

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void log_fp16(const struct LogParams* lParams) {
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

#pragma clang loop unroll_count(8)
    for (i = 0; i < numVectors; i++) {
        half8 vin = p_act_data_v[i];
        half8 vout;
        log_vec(vin, vout);
        p_act_out_v[i] = vout;
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] = __builtin_shave_sau_log_f16_r(p_act_data_s[i]);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
