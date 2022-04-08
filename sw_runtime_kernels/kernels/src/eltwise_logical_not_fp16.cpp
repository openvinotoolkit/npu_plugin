//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <param_logical_not.h>

using namespace sw_params;

#define VECTOR_SIZE (8)
namespace nn {
namespace shave_lib {

extern "C" {

void eltwise_logical_not_fp16(const struct EltwiseParams* lParams) {
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t nElements = 1;

    for (int i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;
    half hOne = static_cast<half>(1.0);
    half hZero = static_cast<half>(0.0);

#pragma clang loop unroll_count(8)
    for (int v = 0; v < numVectors; v++)
        p_act_out_v[v] = (p_act_data_v[v] == hZero) ? hOne : hZero;

    for (int v = numVectors * VECTOR_SIZE; v < nElements; v++)
        p_act_out_s[v] = static_cast<half>((p_act_data_s[v] == hZero) ? hOne : hZero);
}
}
}  // namespace shave_lib
}  // namespace nn
