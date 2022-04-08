//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mv_types.h>
#include <param_negative.h>

using namespace sw_params;

#define VECTOR_SIZE (8)

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void single_shave_negative(uint32_t lParamsAddr) {
    const NegativeParams* lParams = (const NegativeParams*)lParamsAddr;

    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);  // 0x1F000000
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);  // 0x1F004000

    half* p_act_data_s = (half*)(lParams->input.dataAddr);  // 0x1F000000
    half* p_act_out_s = (half*)(lParams->output.dataAddr);  // 0x1F004000

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;

    for (int i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;
    const half negative = (half)-1;

#pragma clang loop unroll_count(8)
    for (int v = 0; v < numVectors; v++)
        p_act_out_v[v] = p_act_data_v[v] * negative;

    for (int v = numVectors * VECTOR_SIZE; v < nElements; v++)
        p_act_out_s[v] = p_act_data_s[v] * negative;
}
}
}  // namespace shave_lib
}  // namespace nn
