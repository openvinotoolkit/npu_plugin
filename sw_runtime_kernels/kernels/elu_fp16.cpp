// {% copyright %}

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <math.h>
#include <param_elu.h>

#define intrinsic_vau_vec(intrinsic, vin, vout) \
    (vout) = intrinsic((vin))
#define intrinsic_sau_vec(intrinsic, vin, vout) \
    (vout)[0] = intrinsic((vin)[0]);        \
    (vout)[1] = intrinsic((vin)[1]);        \
    (vout)[2] = intrinsic((vin)[2]);        \
    (vout)[3] = intrinsic((vin)[3]);        \
    (vout)[4] = intrinsic((vin)[4]);        \
    (vout)[5] = intrinsic((vin)[5]);        \
    (vout)[6] = intrinsic((vin)[6]);        \
    (vout)[7] = intrinsic((vin)[7]);

#ifdef USE_3720_INTSTRUCTIONS
#define exp_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_exp_v8f16_r, vin, vout))
#else
#define exp_vec(vin, vout) {\
    const unsigned short inv_ln2 = 0x3dc6;\
    const half inv_ln2_h = *reinterpret_cast<const half*>(&inv_ln2);\
    const half8 vinv_ln2 = (half8)inv_ln2_h;\
    intrinsic_sau_vec(__builtin_shave_sau_exp2_f16_l_r, vin * vinv_ln2, vout);\
}
#endif

//#define exp2_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_exp2_f16_l_r, vin, vout)
//#define exp2_vec(vin, vout) {(intrinsic_vec(__builtin_shave_vau_exp_v8f16_r, vin, vout))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void elu_fp16(uint32_t lParamsAddr) {
    const EluParams* lParams = (const EluParams*)lParamsAddr;

    half8* p_act_data = (half8*)(lParams->input.dataAddr);  // 0x1F000000
    half8* p_act_out = (half8*)(lParams->output.dataAddr);  // 0x1F004000

    const EluParams* layerParams = reinterpret_cast<const EluParams*>(lParams);
    const half alpha = layerParams->alpha;
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    const half8 one = (half8)1.0f;
    const half8 zero = (half8)0.0f;

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        // TODO: where pointers patch should be???
        // TODO: check overflow
        nElements *= pDims[i];
    }

#pragma clang loop unroll_count(8)
    for (i = 0; i < nElements; i++) {
        half8 min = __builtin_shave_cmu_min_f16_rr_half8(p_act_data[i], zero);
        half8 max = __builtin_shave_cmu_max_f16_rr_half8(p_act_data[i], zero);

        half8 vIn = min/* * vinv_ln2*/;
        half8 vOut;
        exp_vec(vIn, vOut);

        p_act_out[i] = max + alpha * (vOut - one);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
