//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <math.h>
#include <moviVectorTypes.h>
#include <moviVectorUtils.h>
#include <param_sigmoid.h>

#ifdef USE_3720_INTSTRUCTIONS
#define HALF_TYPE half8
#else
#define HALF_TYPE half
#endif
void sigm_fp16_opt(const int32_t nElements, HALF_TYPE* restrict in, HALF_TYPE* restrict out);

void sigmoid_fp16(const struct SigmoidParams* lParams) {
    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);
    int32_t nElements = 1;
    HALF_TYPE* in = (HALF_TYPE*)(lParams->input.dataAddr);
    HALF_TYPE* out = (HALF_TYPE*)(lParams->output.dataAddr);

    for (int32_t i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    sigm_fp16_opt(nElements, in, out);
}

#ifdef USE_4000_INTSTRUCTIONS
HALF_TYPE sau_sigm(half x) {
    HALF_TYPE y;
    asm volatile("sau.sigm %[out] %[in].0" : [ out ] "=r"(y) : [ in ] "r"(x));
    return y;
}
#endif

// main kernel-body in separate function with 'restrict' leaf pointers
// to benefit more Shave compiler optimizations in the unrolled sequence
void sigm_fp16_opt(const int32_t nElements, HALF_TYPE* restrict in, HALF_TYPE* restrict out) {
#ifdef USE_3720_INTSTRUCTIONS
    int32_t nVec = nElements / 8;
    int32_t nScl = nElements - nVec * 8;
    int32_t i = 0;

#pragma clang loop unroll_count(16)
    for (i = 0; i < nVec; ++i) {
        out[i] = __builtin_shave_vau_sigm_v8f16_r(in[i]);
    }

    // Trailing elements
    if (nScl) {
        half8 trail = __builtin_shave_vau_sigm_v8f16_r(in[i]);
        half* restrict tOut = (half*)(out + i);
        // Do only necessary copies
        for (i = 0; i < nScl; ++i) {
            tOut[i] = trail[i];
        }
    }
#else
    half32* restrict dataVec_in = (half32*)in;
    half32* restrict dataVec_out = (half32*)out;

    const uint32_t element_per_vector_op = (sizeof(half32) >> 1);

    const uint32_t vector_iterations = nElements / element_per_vector_op;

#pragma unroll 16
    for (uint32_t e = 0; e < vector_iterations; ++e) {
        dataVec_out[e] = __builtin_shave_vau_sigm_r_half32(dataVec_in[e]);
    }

    const uint32_t vectorised_element_offset = vector_iterations * element_per_vector_op;

    half* restrict data_in = in + vectorised_element_offset;
    half* restrict data_out = out + vectorised_element_offset;
    const uint32_t scalar_iterations = nElements - vectorised_element_offset;

    for (uint32_t e = 0; e < scalar_iterations; ++e) {
        data_out[e] = sau_sigm(data_in[e]);
    }
#endif
}
