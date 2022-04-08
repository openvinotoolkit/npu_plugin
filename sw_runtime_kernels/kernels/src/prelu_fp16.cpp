//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorTypes.h>
#include <mvSubspaces.h>
#include <param_prelu.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

#define NCHW_REV 0x4321  // reverse code of ND_NCHW 0x1234
#define NCWH_REV 0x3421  // reverse code of ND_NCWH 0x1243
#define NHWC_REV 0x2431  // reverse code of ND_NHWC 0x1342
#define NWHC_REV 0x2341  // reverse code of ND_NWHC 0x1432

using namespace sw_params;
namespace nn {
namespace shave_lib {

void prelu_nchw_compute(half* __restrict in, half* __restrict out, half* __restrict slope, int32_t nElementsWH,
                        int32_t nElementsSlope) {
    // TODO: vectorize the implementation
    for (int32_t j = 0; j < nElementsSlope; j++) {
        for (uint32_t i = nElementsWH * j; i < nElementsWH * (j + 1); i++) {
            half min = __builtin_shave_cmu_min_f16_rr_half(in[i], 0.f);
            half max = __builtin_shave_cmu_max_f16_rr_half(in[i], 0.f);

            out[i] = max + in[j] * min;
        }
    }
}

void prelu_NCHW(const struct PReluParams* lParams) {
    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_slope_s = (half*)(lParams->negativeSlope.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t* pDimsSlope = (int32_t*)(lParams->negativeSlope.dimsAddr);

    int32_t nElementsSlope = 1;
    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    for (i = 0; i != lParams->negativeSlope.numDims; i++) {
        nElementsSlope *= pDimsSlope[i];
    }

    int32_t nElementsWH = nElements / nElementsSlope;

    prelu_nchw_compute(p_act_data_s, p_act_out_s, p_act_slope_s, nElementsWH, nElementsSlope);
}

void prelu_nhwc_compute(half* __restrict in, half* __restrict out, half* __restrict slope, int32_t nElementsWH,
                        int32_t nElementsSlope) {
    for (int32_t i = 0; i < nElementsWH; i++) {
        int32_t j = (nElementsSlope * i);
        int32_t k_v = 0;
        const half8 zero = (half8)0.0f;
        half8* in_v = (half8*)(&in[j]);
        half8* out_v = (half8*)(&out[j]);
        half8* slope_v = (half8*)(&slope[0]);
        if (nElementsSlope > VECTOR_SIZE) {
            for (k_v = 0; k_v <= nElementsSlope - VECTOR_SIZE; k_v += VECTOR_SIZE) {
                half8 min_v = __builtin_shave_cmu_min_f16_rr_half8(*in_v, zero);
                half8 max_v = __builtin_shave_cmu_max_f16_rr_half8(*(in_v++), zero);
                *(out_v++) = max_v + *(slope_v++) * min_v;
                j += VECTOR_SIZE;
            }
        }
        for (int32_t k = k_v; k < nElementsSlope; k++) {
            half min = __builtin_shave_cmu_min_f16_rr_half(in[j], 0.f);
            half max = __builtin_shave_cmu_max_f16_rr_half(in[j], 0.f);
            out[j++] = max + slope[k] * min;
        }
    }
}

void prelu_NHWC(const struct PReluParams* lParams) {
    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_slope_s = (half*)(lParams->negativeSlope.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t* pDimsSlope = (int32_t*)(lParams->negativeSlope.dimsAddr);

    int32_t nElementsSlope = 1;
    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    for (i = 0; i != lParams->negativeSlope.numDims; i++) {
        nElementsSlope *= pDimsSlope[i];
    }

    int32_t nElementsWH = nElements / nElementsSlope;

    prelu_nhwc_compute(p_act_data_s, p_act_out_s, p_act_slope_s, nElementsWH, nElementsSlope);
}

extern "C" {
// The algorithm for the PRelu kernel only works for cases where the slope rank is 1 and its dimension is equal to the
// second dim of data input, where then per channel broadcast is applied.
// The case where the slope input should be broadcasted with numpy rules is not supported.
void prelu_fp16(const struct PReluParams* lParams) {
    const PReluParams* p = reinterpret_cast<const PReluParams*>(lParams);

    // If AcrossCh, doing any layout via faster NCHW impl
    if (p->input.dimsOrder == NCWH_REV || p->input.dimsOrder == NCHW_REV) {
        prelu_NCHW(p);
    } else if (p->input.dimsOrder == NHWC_REV || p->input.dimsOrder == NWHC_REV) {
        prelu_NHWC(p);
    }
}
}
}  // namespace shave_lib
}  // namespace nn
