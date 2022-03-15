//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_hswish.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void hswish_fp16(uint32_t lParamsAddr) {
//    int * tmp = (int*)0x2e006580;
//    tmp[1] = 123;
    const HSwishParams* lParams = reinterpret_cast<const HSwishParams*>(lParamsAddr);
//    tmp[2] = (int)lParams;

    half* __restrict__ p_act_data = reinterpret_cast<half*>(lParams->input.dataAddr);  // 0x1F000000
    half* __restrict__ p_act_out = reinterpret_cast<half*>(lParams->output.dataAddr);  // 0x1F004000
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
//    tmp[3] = (int)p_act_data;
//    tmp[4] = (int)p_act_out;
//    tmp[5] = (int)pDims;

    int32_t nElements = 1;
    for (int i = 0; i != lParams->input.numDims; i++) {
        // TODO: check overflow
        nElements *= pDims[i];
//        tmp[6+i] = (int)pDims[i];
    }
//    tmp[6+lParams->input.numDims] = nElements;
//
//    ((float*)tmp)[10+0] = 1.f;
//    tmp[10+1] = ((int*)p_act_data)[0];
//    tmp[10+2] = ((int*)p_act_data)[1];
//    tmp[10+3] = ((int*)p_act_data)[2];
//    tmp[10+1] = float(p_act_data[0]);
//    tmp[10+2] = float(p_act_data[1]);
//    tmp[10+3] = float(p_act_data[2]);

    int32_t j = 0;
    const float max_val_6 = 6.0f;
    const float inv_max_val_6 = 1.f/max_val_6;
#ifndef __shavenn_ISA__
    const half8 hmax_val_6 = max_val_6;
    const half8 add_val_3 = 3.0f;
    for(; j < nElements - (VECTOR_SIZE - 1); j += VECTOR_SIZE){
        half8 ri = ((half8*)(p_act_data + j))[0];
        ((half8*)(p_act_out + j))[0] =  __builtin_shave_cmu_clamp0_f16_rr_half8(ri + add_val_3, hmax_val_6) * ri * inv_max_val_6;
    }
#endif

    for(; j < nElements; j++){
        p_act_out[j] = p_act_data[j] * MIN(6.f, MAX(0.f, p_act_data[j] + 3.f)) * inv_max_val_6;
//        p_act_out[j] = p_act_data[j];//(float)i;
//        tmp[10+i] = float(p_act_data[i]);
//        tmp[20+i] = float(p_act_out[i]);
    }
//    ((float*)tmp)[20+0] = 2.f;
//    tmp[20+1] = ((int*)p_act_out)[0];
//    tmp[20+2] = ((int*)p_act_out)[1];
//    tmp[20+3] = ((int*)p_act_out)[2];
}

}
} // namespace shave_lib
} // namespace nn
