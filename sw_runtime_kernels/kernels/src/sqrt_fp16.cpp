//
// Copyright Intel Corporation.
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
#include <param_sqrt.h>

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
#define sqrt_vec(vin, vout) (intrinsic_vau_vec(__builtin_shave_vau_sqt_v8f16_r, vin, vout))
#else
#define sqrt_vec(vin, vout) \
    { intrinsic_sau_vec(__builtin_shave_sau_sqt_f16_l_r, vin, vout); }
#endif

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void sqrt_fp16(const struct SqrtParams* lParams) {
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
        sqrt_vec(vin, vout);
        p_act_out_v[i] = vout;
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] = __builtin_shave_sau_sqt_f16_l_r(p_act_data_s[i]);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
