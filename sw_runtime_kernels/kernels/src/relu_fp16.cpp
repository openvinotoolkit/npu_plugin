//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_relu.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void relu_fp16(uint32_t lParamsAddr) {
    const ReluParams* lParams = reinterpret_cast<const ReluParams*>(lParamsAddr);

    half8* p_act_data_v = (half8*)(lParams->input.dataAddr); // 0x1F000000
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr); // 0x1F004000

    half* p_act_data_s = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out_s = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    const half8 zero = 0.0f;

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        // TODO: check overflow
        nElements *= pDims[i];
    }

    const int numVectors = nElements / VECTOR_SIZE;

#pragma clang loop unroll_count(8)
    for (i = 0; i < numVectors; ++i) {
        p_act_out_v[i] =  __builtin_shave_cmu_max_f16_rr_half8(p_act_data_v[i], zero);
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] = __builtin_shave_cmu_max_f16_rr_half(p_act_data_s[i], 0.f);
    }
}

}
} // namespace shave_lib
} // namespace nn
