// {% copyright %}

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

#define UNROLL_SIZE 8  // Changes to this should be reflected in the code as well.

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void hswish_fp16(uint32_t lParamsAddr) {
    const HSwishParams* lParams = reinterpret_cast<const HSwishParams*>(lParamsAddr);

    half8* __restrict__ p_act_data = reinterpret_cast<half8*>(lParams->input.dataAddr);  // 0x1F000000
    half8* __restrict__ p_act_out = reinterpret_cast<half8*>(lParams->output.dataAddr);  // 0x1F004000

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    const half8 add_val3 = 3.0f;
    const half8 max_val_6 = 6.0f;

    int32_t nElements = 1;
    int32_t i = 0;
    for (i = 0; i != lParams->input.numDims; i++) {
        // TODO: check overflow
        nElements *= pDims[i];
    }

    for (i = 0; i < ((nElements / UNROLL_SIZE) * UNROLL_SIZE); i += UNROLL_SIZE) {
        p_act_out[i + 0] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 0] + add_val3, max_val_6) * p_act_data[i + 0] / max_val_6;
        p_act_out[i + 1] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 1] + add_val3, max_val_6) * p_act_data[i + 1] / max_val_6;
        p_act_out[i + 2] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 2] + add_val3, max_val_6) * p_act_data[i + 2] / max_val_6;
        p_act_out[i + 3] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 3] + add_val3, max_val_6) * p_act_data[i + 3] / max_val_6;
        p_act_out[i + 4] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 4] + add_val3, max_val_6) * p_act_data[i + 4] / max_val_6;
        p_act_out[i + 5] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 5] + add_val3, max_val_6) * p_act_data[i + 5] / max_val_6;
        p_act_out[i + 6] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 6] + add_val3, max_val_6) * p_act_data[i + 6] / max_val_6;
        p_act_out[i + 7] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i + 7] + add_val3, max_val_6) * p_act_data[i + 7] / max_val_6;
    }

    for(; i < nElements; ++i){
        p_act_out[i] =  __builtin_shave_cmu_clamp0_f16_rr_half8(p_act_data[i] + add_val3, max_val_6) * p_act_data[i] / max_val_6;
    }
}

}
} // namespace shave_lib
} // namespace nn
