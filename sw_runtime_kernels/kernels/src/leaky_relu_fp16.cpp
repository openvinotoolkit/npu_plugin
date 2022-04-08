//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <param_leaky_relu.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void leaky_relu_fp16(const struct LeakyReluParams* lParams) {
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    const half p_act_slope = (half)(lParams->negative_slope);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    const half8 zero = (half8)0.0f;
    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;

    for (i = 0; i < numVectors; i++) {
        half8 min = __builtin_shave_cmu_min_f16_rr_half8(p_act_data_v[i], zero);
        half8 max = __builtin_shave_cmu_max_f16_rr_half8(p_act_data_v[i], zero);

        p_act_out_v[i] = max + p_act_slope * min;
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        half min = __builtin_shave_cmu_min_f16_rr_half(p_act_data_s[i], 0.f);
        half max = __builtin_shave_cmu_max_f16_rr_half(p_act_data_s[i], 0.f);

        p_act_out_s[i] = max + p_act_slope * min;
    }
}
}
}  // namespace shave_lib
}  // namespace nn
