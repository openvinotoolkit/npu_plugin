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
#include <param_maxmin.h>
#include <math.h>

#define VECTOR_SIZE 8  // Changes to this should be reflected in the code as well.

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void maximum(const struct MaxMinParams *lParams) {
    half8* in_v = (half8*)lParams->input.dataAddr;
    half8* in2_v = (half8*)lParams->input2.dataAddr;
    half8* out_v = (half8*)lParams->output.dataAddr;

    half* in_s = (half*)lParams->input.dataAddr;
    half* in2_s = (half*)lParams->input2.dataAddr;
    half* out_s = (half*)lParams->output.dataAddr;

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);
    int32_t nElements = 1;

    for (int32_t i = 0; i != lParams->input.numDims; i++) {
        nElements *=  pDims[i];
    }

    int32_t numVectors = floor(nElements / VECTOR_SIZE);

    for (int32_t i = 0; i < numVectors; i++) {
        out_v[i] = __builtin_shave_cmu_max_f16_rr_half8(in_v[i], in2_v[i]);
    }

    // Compensate
    for (int32_t i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        out_s[i] = __builtin_shave_cmu_max_f16_rr_half(in_s[i], in2_s[i]);
    }
}

void minimum(const struct MaxMinParams *lParams) {
    half8* in_v = (half8*)lParams->input.dataAddr;
    half8* in2_v = (half8*)lParams->input2.dataAddr;
    half8* out_v = (half8*)lParams->output.dataAddr;

    half* in_s = (half*)lParams->input.dataAddr;
    half* in2_s = (half*)lParams->input2.dataAddr;
    half* out_s = (half*)lParams->output.dataAddr;

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);
    int32_t nElements = 1;

    for (int32_t i = 0; i != lParams->input.numDims; i++) {
        nElements *=  pDims[i];
    }

    int32_t numVectors = floor(nElements / VECTOR_SIZE);

    for (int32_t i = 0; i < numVectors; i++) {
        out_v[i] = __builtin_shave_cmu_min_f16_rr_half8(in_v[i], in2_v[i]);
    }

    // Compensate
    for (int32_t i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        out_s[i] = __builtin_shave_cmu_min_f16_rr_half(in_s[i], in2_s[i]);
    }
}

}
}  // namespace shave_lib
}  // namespace nn
