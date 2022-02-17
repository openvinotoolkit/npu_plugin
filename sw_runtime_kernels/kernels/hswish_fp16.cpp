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
#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_nn.h>
#else
#include <dma_shave.h>
#endif

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void hswish_fp16(uint32_t lParamsAddr) {
    const HSwishParams* lParams = reinterpret_cast<const HSwishParams*>(lParamsAddr);

    half* __restrict__ p_act_data = reinterpret_cast<half*>(lParams->input.dataAddr);  // 0x1F000000
    half* __restrict__ p_act_out = reinterpret_cast<half*>(lParams->output.dataAddr);  // 0x1F004000
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    for (int i = 0; i != lParams->input.numDims; i++) {
        // TODO: check overflow
        nElements *= pDims[i];
    }

    int32_t i = 0;
    const float max_val_6 = 6.0f;
    const float inv_max_val_6 = 1.f/max_val_6;
#ifndef __shavenn_ISA__
    const half8 hmax_val_6 = max_val_6;
    const half8 add_val_3 = 3.0f;
    for(; i < nElements - (VECTOR_SIZE - 1); i += VECTOR_SIZE){
        half8 ri = ((half8*)(p_act_data + i))[0];
        ((half8*)(p_act_out + i))[0] =  __builtin_shave_cmu_clamp0_f16_rr_half8(ri + add_val_3, hmax_val_6) * ri * inv_max_val_6;
    }
#endif

    for(; i < nElements; i++){
        p_act_out[i] = p_act_data[i] * MIN(6.f, MAX(0.f, p_act_data[i] + 3.f)) * inv_max_val_6;
    }
}

}
} // namespace shave_lib
} // namespace nn
