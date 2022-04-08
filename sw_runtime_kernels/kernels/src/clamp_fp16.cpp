//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_clamp.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void clamp_fp16(const struct ClampParams* lParams) {
    half8* p_act_in_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_in_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    const half8 minVal = (half8)(lParams->min);
    const half8 maxVal = (half8)(lParams->max);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t j = 0;
    for (j = 0; j != lParams->input.numDims; j++) {
        nElements *= pDims[j];
    }
    const int numVectors = nElements / VECTOR_SIZE;
    for (int i = 0; i < numVectors; ++i) {
        *p_act_out_v++ = __builtin_shave_cmu_clampab_f16_rrr_half8(*p_act_in_v++, minVal, maxVal);
    }
    for (int i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] = __builtin_shave_cmu_clampab_f16_rrr_half(p_act_in_s[i], lParams->min, lParams->max);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
