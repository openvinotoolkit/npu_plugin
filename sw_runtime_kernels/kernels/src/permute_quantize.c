//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <math.h>
#include <moviVectorConvert.h>
#include <moviVectorTypes.h>
#include <param_permute_quantize.h>
#include <stdint.h>

// #define USE_C_VERSION

void getCoord(int nSubspace, const int32_t dims[], int nDims, int32_t subspaceCoord[]) {
    for (int i = 0; i < nDims; ++i) {
        int nUpSubspace = nSubspace / dims[i];
        subspaceCoord[i] = nSubspace - nUpSubspace * dims[i];
        nSubspace = nUpSubspace;
    }
}

void increment1Coord(int32_t subspaceCoord[], const int32_t dims[], int nDims) {
    for (int d = 0; d < nDims; ++d) {
        if (subspaceCoord[d] < dims[d] - 1) {
            subspaceCoord[d]++;
            return;
        }
        subspaceCoord[d] = 0;
    }
}

int getOffsetU8(const int32_t subspaceCoord[], const int32_t strides[], int nDims) {
    int offset = 0;
    for (int d = 0; d < nDims; ++d) {
        const int coord = subspaceCoord[d];
        offset += coord * strides[d];
    }
    return offset;
}

void permuteArray(const int32_t src_set[], const int32_t permutation[], int32_t dst_set[], int set_lng) {
    for (int i = 0; i < set_lng; i++) {
        dst_set[i] = src_set[permutation[i]];
    }
}

void __attribute__((noinline)) pqNchwNhwcC1_asm(uchar8* __restrict out,            // i18
                                                half8* __restrict in0,             // i17
                                                const half scaleFact,              // i16
                                                const half zeroFact,               // i15
                                                const uint32_t runSz,              // i14
                                                const uint32_t strideC);           // i13
void __attribute__((noinline)) pqNchwNhwcC1_asm_stride(uchar8* __restrict out,     // i18
                                                       half8* __restrict in0,      // i17
                                                       const half scaleFact,       // i16
                                                       const half zeroFact,        // i15
                                                       const uint32_t runSz,       // i14
                                                       const uint32_t strideC);    // i13
void __attribute__((noinline)) pqNchwNhwcC1Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          const half scaleFact,    // i16
                                                          const half zeroFact,     // i15
                                                          const uint32_t runSz);   // i14
static void pqNchwNhwcC1(uchar* out, half8* in0, const float scaleFact, const float zeroFact, const uint32_t runSz,
                         const uint32_t strideC) {
    const uint32_t nVec = runSz >> 3;
    const float8 scale = scaleFact;
    const float8 zero = zeroFact;
    const half8 cmin = (half8)0.0h;
    const half8 cmax = (half8)255.0h;
    uint32_t xo = 0;
    for (uint32_t x = 0; x < nVec; x++) {
        float8 fi0 = mvuConvert_float8(in0[x]);
        float8 qi0 = fi0 * scale + zero;
        half8 dd = mvuConvert_half8(qi0);
        uchar8 odd = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd, cmin, cmax));
        out[xo] = odd[0];
        xo += strideC;
        out[xo] = odd[1];
        xo += strideC;
        out[xo] = odd[2];
        xo += strideC;
        out[xo] = odd[3];
        xo += strideC;
        out[xo] = odd[4];
        xo += strideC;
        out[xo] = odd[5];
        xo += strideC;
        out[xo] = odd[6];
        xo += strideC;
        out[xo] = odd[7];
        xo += strideC;
    }
}

void __attribute__((noinline)) pqNchwNhwcC3_asm(uchar8* __restrict out,   // i18
                                                half8* __restrict in0,    // i17
                                                half8* __restrict in1,    // i16
                                                half8* __restrict in2,    // i15
                                                const half scaleFact,     // i14
                                                const half zeroFact,      // i13
                                                const uint32_t runSz,     // i12
                                                const uint32_t strideC);  // i11

void __attribute__((noinline)) pqNchwNhwcC3Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          half8* __restrict in1,   // i16
                                                          half8* __restrict in2,   // i15
                                                          const half scaleFact,    // i14
                                                          const half zeroFact,     // i13
                                                          const uint32_t runSz);   // i12

static void pqNchwNhwcC3(uchar* out, half8* in0, half8* in1, half8* in2, const float scaleFact, const float zeroFact,
                         const uint32_t runSz, const uint32_t strideC) {
    const uint32_t nVec = runSz >> 3;
    const float8 scale = scaleFact;
    const float8 zero = zeroFact;
    const half8 cmin = (half8)0.0h;
    const half8 cmax = (half8)255.0h;
    uint32_t xo = 0;
    for (uint32_t x = 0; x < nVec; x++) {
        float8 fi0 = mvuConvert_float8(in0[x]);
        float8 fi1 = mvuConvert_float8(in1[x]);
        float8 fi2 = mvuConvert_float8(in2[x]);

        float8 qi0 = fi0 * scale + zero;
        float8 qi1 = fi1 * scale + zero;
        float8 qi2 = fi2 * scale + zero;

        float8 pi0 = {qi0[0], qi0[1], qi0[2], qi0[3], qi0[4], qi0[5], qi0[6], qi0[7]};
        float8 pi1 = {qi1[0], qi1[1], qi1[2], qi1[3], qi1[4], qi1[5], qi1[6], qi1[7]};
        float8 pi2 = {qi2[0], qi2[1], qi2[2], qi2[3], qi2[4], qi2[5], qi2[6], qi2[7]};

        half8 dd0 = mvuConvert_half8(pi0);
        half8 dd1 = mvuConvert_half8(pi1);
        half8 dd2 = mvuConvert_half8(pi2);

        uchar8 odd0 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd0, cmin, cmax));
        uchar8 odd1 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd1, cmin, cmax));
        uchar8 odd2 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd2, cmin, cmax));

        out[xo + 0] = odd0[0];
        out[xo + 1] = odd1[0];
        out[xo + 2] = odd2[0];
        xo += strideC;

        out[xo + 0] = odd0[1];
        out[xo + 1] = odd1[1];
        out[xo + 2] = odd2[1];
        xo += strideC;

        out[xo + 0] = odd0[2];
        out[xo + 1] = odd1[2];
        out[xo + 2] = odd2[2];
        xo += strideC;

        out[xo + 0] = odd0[3];
        out[xo + 1] = odd1[3];
        out[xo + 2] = odd2[3];
        xo += strideC;

        out[xo + 0] = odd0[4];
        out[xo + 1] = odd1[4];
        out[xo + 2] = odd2[4];
        xo += strideC;

        out[xo + 0] = odd0[5];
        out[xo + 1] = odd1[5];
        out[xo + 2] = odd2[5];
        xo += strideC;

        out[xo + 0] = odd0[6];
        out[xo + 1] = odd1[6];
        out[xo + 2] = odd2[6];
        xo += strideC;

        out[xo + 0] = odd0[7];
        out[xo + 1] = odd1[7];
        out[xo + 2] = odd2[7];
        xo += strideC;
    }
}

void __attribute__((noinline)) pqNchwNhwcC4_asm(uchar8* __restrict out,            // i18
                                                half8* __restrict in0,             // i17
                                                half8* __restrict in1,             // i16
                                                half8* __restrict in2,             // i15
                                                half8* __restrict in3,             // i14
                                                const half scaleFact,              // i13
                                                const half zeroFact,               // i12
                                                const uint32_t runSz,              // i11
                                                const uint32_t strideC);           // i19
void __attribute__((noinline)) pqNchwNhwcC4Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          half8* __restrict in1,   // i16
                                                          half8* __restrict in2,   // i15
                                                          half8* __restrict in3,   // i14
                                                          const half scaleFact,    // i13
                                                          const half zeroFact,     // i12
                                                          const uint32_t runSz);   // i11

static void pqNchwNhwcC4(uchar* out, half8* in0, half8* in1, half8* in2, half8* in3, const float scaleFact,
                         const float zeroFact, const uint32_t runSz, const uint32_t strideC) {
    const uint32_t nVec = runSz >> 3;
    const float8 scale = scaleFact;
    const float8 zero = zeroFact;
    const half8 cmin = (half8)0.0h;
    const half8 cmax = (half8)255.0h;
    uint32_t xo = 0;
    for (uint32_t x = 0; x < nVec; x++) {
        float8 fi0 = mvuConvert_float8(in0[x]);
        float8 fi1 = mvuConvert_float8(in1[x]);
        float8 fi2 = mvuConvert_float8(in2[x]);
        float8 fi3 = mvuConvert_float8(in3[x]);

        float8 qi0 = fi0 * scale + zero;
        float8 qi1 = fi1 * scale + zero;
        float8 qi2 = fi2 * scale + zero;
        float8 qi3 = fi3 * scale + zero;

        float8 pi0 = {qi0[0], qi0[1], qi0[2], qi0[3], qi0[4], qi0[5], qi0[6], qi0[7]};
        float8 pi1 = {qi1[0], qi1[1], qi1[2], qi1[3], qi1[4], qi1[5], qi1[6], qi1[7]};
        float8 pi2 = {qi2[0], qi2[1], qi2[2], qi2[3], qi2[4], qi2[5], qi2[6], qi2[7]};
        float8 pi3 = {qi3[0], qi3[1], qi3[2], qi3[3], qi3[4], qi3[5], qi3[6], qi3[7]};

        half8 dd0 = mvuConvert_half8(pi0);
        half8 dd1 = mvuConvert_half8(pi1);
        half8 dd2 = mvuConvert_half8(pi2);
        half8 dd3 = mvuConvert_half8(pi3);

        uchar8 odd0 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd0, cmin, cmax));
        uchar8 odd1 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd1, cmin, cmax));
        uchar8 odd2 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd2, cmin, cmax));
        uchar8 odd3 = mvuConvert_uchar8(__builtin_shave_cmu_clampab_f16_rrr_half8(dd3, cmin, cmax));

        out[xo + 0] = odd0[0];
        out[xo + 1] = odd1[0];
        out[xo + 2] = odd2[0];
        out[xo + 3] = odd3[0];
        xo += strideC;

        out[xo + 0] = odd0[1];
        out[xo + 1] = odd1[1];
        out[xo + 2] = odd2[1];
        out[xo + 3] = odd3[1];
        xo += strideC;

        out[xo + 0] = odd0[2];
        out[xo + 1] = odd1[2];
        out[xo + 2] = odd2[2];
        out[xo + 3] = odd3[2];
        xo += strideC;

        out[xo + 0] = odd0[3];
        out[xo + 1] = odd1[3];
        out[xo + 2] = odd2[3];
        out[xo + 3] = odd3[3];
        xo += strideC;

        out[xo + 0] = odd0[4];
        out[xo + 1] = odd1[4];
        out[xo + 2] = odd2[4];
        out[xo + 3] = odd3[4];
        xo += strideC;

        out[xo + 0] = odd0[5];
        out[xo + 1] = odd1[5];
        out[xo + 2] = odd2[5];
        out[xo + 3] = odd3[5];
        xo += strideC;

        out[xo + 0] = odd0[6];
        out[xo + 1] = odd1[6];
        out[xo + 2] = odd2[6];
        out[xo + 3] = odd3[6];
        xo += strideC;

        out[xo + 0] = odd0[7];
        out[xo + 1] = odd1[7];
        out[xo + 2] = odd2[7];
        out[xo + 3] = odd3[7];
        xo += strideC;
    }
}

void permute_quantize(const struct PermuteQuantizeParams* lParams) {
    int32_t mode = (int32_t)lParams->opt_mode;
    if (PQ_NCHW_NHWC_C3EXP4 == mode) {  // spped up specific usecase
        const half* inData = (const half*)(lParams->input.dataAddr);
        uint8_t* outData = (uint8_t*)(lParams->output.dataAddr);
        const int32_t* inDims = (int32_t*)(lParams->input.dimsAddr);
        uint32_t runSz = inDims[1] * inDims[0];
        const half scaleFact = (half)(1.0f / lParams->scale);
        const half zeroFact = (half)((float)(int32_t)lParams->zero + 0.5f);
        const half* ind0 = &inData[0];
        const half* ind1 = &inData[runSz];
        const half* ind2 = &inData[runSz << 1];
        pqNchwNhwcC3Exp4Algn16_asm((uchar8*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, scaleFact, zeroFact,
                                   runSz);
    } else {
        const half* inData = (const half*)(lParams->input.dataAddr);
        uint8_t* outData = (uint8_t*)(lParams->output.dataAddr);
        const int32_t* inDims = (int32_t*)(lParams->input.dimsAddr);
        const int32_t* outDims = (int32_t*)(lParams->output.dimsAddr);
        const uint32_t numScales = 1;
        float scales[1];
        int32_t zeroes[1];
        scales[0] = lParams->scale;
        zeroes[0] = (int32_t)lParams->zero;
        uint32_t strideC = outDims[0];
        int32_t mode = (int32_t)lParams->opt_mode;
        switch (mode) {
        case PQ_NCHW_NHWC_C1: {
            uint32_t runSz = inDims[1] * inDims[0];
            const half scaleFact = (half)(1.0 / scales[0]);
            const half zeroFact = (half)(zeroes[0] + 0.5f);
            uint32_t vectSize = (runSz & (~0x1F));
            if (vectSize) {
#ifdef USE_C_VERSION  // C reference
                pqNchwNhwcC1((uchar*)outData, (half8*)inData, scaleFact, zeroFact, vectSize, strideC);
#else  // ASM opt
                if (1 == strideC) {
                    pqNchwNhwcC1_asm((uchar8*)outData, (half8*)inData, scaleFact, zeroFact, vectSize, strideC);
                } else {
                    pqNchwNhwcC1_asm_stride((uchar8*)outData, (half8*)inData, scaleFact, zeroFact, vectSize, strideC);
                }
#endif
            }
            // Trailing elements
            if ((runSz & (0x7)) > 0) {
                const uint32_t shiftVec = runSz & (~0x1F);
                uint32_t xo = shiftVec * strideC;
                for (uint32_t x = shiftVec; x < runSz; x++) {
                    outData[xo] = (uchar)(inData[x] * scaleFact + zeroFact);
                    xo += strideC;
                }
            }

        } break;
        case PQ_NCHW_NHWC_C3: {
            uint32_t runSz = inDims[1] * inDims[0];
            const half scaleFact = (half)(1.0f / scales[0]);
            const half zeroFact = (half)(zeroes[0] + 0.5f);
            const half* ind0 = &inData[0];
            const half* ind1 = &inData[runSz];
            const half* ind2 = &inData[runSz << 1];
            uint32_t vectSize = (runSz & (~0x7));
            if (vectSize) {
#ifdef USE_C_VERSION  // C reference
                pqNchwNhwcC3((uchar*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, scaleFact, zeroFact, vectSize,
                             strideC);
#else  // ASM opt
                pqNchwNhwcC3_asm((uchar8*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, scaleFact, zeroFact,
                                 vectSize, strideC);
#endif
            }
            // Trailing elements
            if ((runSz & (0x7)) > 0) {
                const uint32_t shiftVec = runSz & (~0x7);
                uint32_t xo = shiftVec * strideC;
                for (uint32_t x = shiftVec; x < runSz; x++) {
                    outData[xo] = (uchar)(ind0[x] * scaleFact + zeroFact);
                    outData[xo + 1] = (uchar)(ind1[x] * scaleFact + zeroFact);
                    outData[xo + 2] = (uchar)(ind2[x] * scaleFact + zeroFact);
                    xo += strideC;
                }
            }
        } break;
        case PQ_NCHW_NHWC_C4: {
            uint32_t runSz = inDims[1] * inDims[0];
            const half scaleFact = (half)(1.0f / scales[0]);
            const half zeroFact = (half)(zeroes[0] + 0.5f);
            const half* ind0 = &inData[0];
            const half* ind1 = &inData[runSz];
            const half* ind2 = &inData[runSz << 1];
            const half* ind3 = &inData[runSz * 3];
            uint32_t vectSize = (runSz & (~0x7));
            if (vectSize) {
#ifdef USE_C_VERSION  // C reference
                pqNchwNhwcC4((uchar*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, (half8*)ind3, scaleFact,
                             zeroFact, vectSize, strideC);
#else  // ASM opt
                pqNchwNhwcC4_asm((uchar8*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, (half8*)ind3, scaleFact,
                                 zeroFact, vectSize, strideC);
#endif
            }
            // Trailing elements
            if ((runSz & (0x7)) > 0) {
                const uint32_t shiftVec = runSz & (~0x7);
                uint32_t xo = shiftVec * strideC;
                for (uint32_t x = shiftVec; x < runSz; x++) {
                    outData[xo] = (uchar)(ind0[x] * scaleFact + zeroFact);
                    outData[xo + 1] = (uchar)(ind1[x] * scaleFact + zeroFact);
                    outData[xo + 2] = (uchar)(ind2[x] * scaleFact + zeroFact);
                    outData[xo + 3] = (uchar)(ind3[x] * scaleFact + zeroFact);
                    xo += strideC;
                }
            }
        } break;

        case PQ_NCHW_NHWC_C4EXP4: {
            uint32_t runSz = inDims[1] * inDims[0];
            const half scaleFact = (half)(1.0f / scales[0]);
            const half zeroFact = (half)(zeroes[0] + 0.5f);
            const half* ind0 = &inData[0];
            const half* ind1 = &inData[runSz];
            const half* ind2 = &inData[runSz << 1];
            const half* ind3 = &inData[runSz * 3];
#ifdef USE_C_VERSION  // C reference
            pqNchwNhwcC4((uchar*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, (half8*)ind3, scaleFact, zeroFact,
                         runSz, strideC);
#else  // ASM opt
            pqNchwNhwcC4Exp4Algn16_asm((uchar8*)outData, (half8*)ind0, (half8*)ind1, (half8*)ind2, (half8*)ind3,
                                       scaleFact, zeroFact, runSz);
#endif
        } break;
        case PQ_NCHW_NHWC_C1EXP4: {
            uint32_t runSz = inDims[1] * inDims[0];
            const half scaleFact = (half)(1.0 / scales[0]);
            const half zeroFact = (half)(zeroes[0] + 0.5f);
#ifdef USE_C_VERSION  // C reference
            pqNchwNhwcC1((uchar*)outData, (half8*)inData, scaleFact, zeroFact, runSz, strideC);
#else  // ASM opt
            pqNchwNhwcC1Exp4Algn16_asm((uchar8*)outData, (half8*)inData, scaleFact, zeroFact, runSz);

#endif
        } break;

        default: {
            const int ndims = lParams->input.numDims;
            //  Get Permutation params
            const int64_t* perm64 = (int64_t*)(lParams->perm);

            // Get strides
            const uint64_t* inStrides64 = (uint64_t*)(lParams->input.stridesAddr);
            const uint64_t* outStrides64 = (uint64_t*)(lParams->output.stridesAddr);

            int32_t inStrides[MAX_ND_DIMS] = {};
            int32_t outStrides[MAX_ND_DIMS] = {};
            int32_t perm[MAX_ND_DIMS] = {};

            for (int i = 0; i < ndims; ++i) {
                inStrides[i] = (int32_t)(inStrides64[i] / 8);
                outStrides[i] = (int32_t)(outStrides64[i] / 8);
                perm[i] = (int32_t)(perm64[i]);
                if (perm[i] == 2)
                    strideC = outDims[i];
            }

            // Get number of elements
            int total = 1;
            for (int i = 0; i < ndims; i++) {
                total *= inDims[i];
            }

            int spatial_total = total / numScales;
            // Start kernel
            int32_t in[MAX_ND_DIMS] = {};
            getCoord(0, inDims, ndims, in);
            for (int current = 0; current < total; ++current) {
                int32_t out[MAX_ND_DIMS] = {};
                permuteArray(in, perm, out, ndims);

                int inOfs = getOffsetU8(in, inStrides, ndims) / sizeof(half);
                int outOfs = getOffsetU8(out, outStrides, ndims);

                int index = numScales > 1 ? outOfs / spatial_total : 0;

                outData[outOfs] = (uchar)(inData[inOfs] / scales[index] + zeroes[index] + 0.5h);

                increment1Coord(in, inDims, ndims);
            }
        } break;
        }
    }
}

void __attribute__((noinline)) pqNchwNhwcC4_asm(uchar8* __restrict out,    // i18
                                                half8* __restrict in0,     // i17
                                                half8* __restrict in1,     // i16
                                                half8* __restrict in2,     // i15
                                                half8* __restrict in3,     // i14
                                                const half scaleFact,      // i13
                                                const half zeroFact,       // i12
                                                const uint32_t runSz,      // i11
                                                const uint32_t strideC) {  // i19
    __asm volatile(
            "lsu0.ld.32 i1, i19  \n"
            "cmu.cp.128.16.r v0, i13.0 \n"
            "cmu.cp.128.16.r v1, i12.0 \n"
            "lsu0.ldi.128   v2, i17 || iau.incs i11, -8 \n"
            "lsu1.ldi.128   v3, i16 || peu.pcix LTE 0 || bru.bra pqNchwNhwcC4_endloop0 \n"
            "lsu0.ldi.128   v4, i15 \n"
            "lsu1.ldi.128   v5, i14 \n"
            "nop 2 \n"
            "iau.mul i8, i1, 0x4 \n"
            "CMU.CP.32 I5, I18 \n"
            "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 || iau.incs i11, -8 \n"
            "vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 || peu.pcix LTE 0 || bru.bra pqNchwNhwcC4_endloop1 \n"
            "vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 \n"
            "vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i14 \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1 \n"
            "vau.add.f16 v13, v5, v1 \n"
            "pqNchwNhwcC4_loop: \n"
            "CMU.CP.64.F16.U8S V6.0, V10 || vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
            "CMU.CP.64.F16.U8S V7.0, V11 || vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 || iau.add i4, I5, 0 \n"
            "CMU.CP.64.F16.U8S V8.0, V12 || vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 || iau.add I5, I4, i8 \n"
            "CMU.CP.64.F16.U8S V9.0, V13 || vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i14  \n"
            "iau.incs i11, -8 \n"
            "cmu.vilv.x8 v10, v12, v7, v6    || vau.add.f16 v10, v2, v1 || peu.pcix GT 0 || bru.bra pqNchwNhwcC4_loop "
            "\n"
            "cmu.vilv.x8 v11, v12, v9, v8    || vau.add.f16 v11, v3, v1  \n"
            "cmu.vilv.x16 v14, v15, v11, v10 || vau.add.f16 v12, v4, v1  \n"
            "lsu0.sti.32 v14.0, i4, i1 || lsu1.sti.32 v15.0, i5, i1 || vau.add.f16 v13, v5, v1 \n"
            "lsu0.sti.32 v14.1, i4, i1 || lsu1.sti.32 v15.1, i5, i1 \n"
            "lsu0.sti.32 v14.2, i4, i1 || lsu1.sti.32 v15.2, i5, i1 \n"
            "lsu0.sti.32 v14.3, i4, i1 || lsu1.sti.32 v15.3, i5, i1 \n"
            "pqNchwNhwcC4_endloop1: \n"
            "CMU.CP.64.F16.U8S V6.0, V10 \n"
            "CMU.CP.64.F16.U8S V7.0, V11 \n"
            "CMU.CP.64.F16.U8S V8.0, V12 \n"
            "CMU.CP.64.F16.U8S V9.0, V13 \n"
            "nop \n"
            "cmu.vilv.x8 v10, v12, v7, v6 || iau.add i4, I5, 0 \n"
            "cmu.vilv.x8 v11, v12, v9, v8 || iau.add I5, I4, i8 \n"
            "cmu.vilv.x16 v14, v15, v11, v10 \n"
            "lsu0.sti.32 v14.0, i4, i1 || lsu1.sti.32 v15.0, i5, i1 \n"
            "lsu0.sti.32 v14.1, i4, i1 || lsu1.sti.32 v15.1, i5, i1 \n"
            "lsu0.sti.32 v14.2, i4, i1 || lsu1.sti.32 v15.2, i5, i1 \n"
            "lsu0.sti.32 v14.3, i4, i1 || lsu1.sti.32 v15.3, i5, i1 \n"
            "pqNchwNhwcC4_endloop0: \n"
            "vau.mul.f16  v2, v2, v0 \n"
            "vau.mul.f16  v3, v3, v0  \n"
            "vau.mul.f16  v4, v4, v0  \n"
            "vau.mul.f16  v5, v5, v0  \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1 \n"
            "vau.add.f16 v13, v5, v1 \n"
            "CMU.CP.64.F16.U8S V6.0, V10 \n"
            "CMU.CP.64.F16.U8S V7.0, V11 \n"
            "CMU.CP.64.F16.U8S V8.0, V12 \n"
            "CMU.CP.64.F16.U8S V9.0, V13 \n"
            "nop  \n"
            "cmu.vilv.x8 v10, v12, v7, v6 || iau.add i4, I5, 0 \n"
            "cmu.vilv.x8 v11, v12, v9, v8 || iau.add I5, I4, i8 \n"
            "cmu.vilv.x16 v14, v15, v11, v10 || BRU.jmp i30 \n"
            "lsu0.sti.32 v14.0, i4, i1 || lsu1.sti.32 v15.0, i5, i1 \n"
            "lsu0.sti.32 v14.1, i4, i1 || lsu1.sti.32 v15.1, i5, i1 \n"
            "lsu0.sti.32 v14.2, i4, i1 || lsu1.sti.32 v15.2, i5, i1 \n"
            "lsu0.sti.32 v14.3, i4, i1 || lsu1.sti.32 v15.3, i5, i1 \n"
            "nop 3 \n"
            :
            : "r"(out), "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(scaleFact), "r"(zeroFact), "r"(runSz), "r"(strideC)
            : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC3_asm(uchar8* __restrict out,    // i18
                                                half8* __restrict in0,     // i17
                                                half8* __restrict in1,     // i16
                                                half8* __restrict in2,     // i15
                                                const half scaleFact,      // i14
                                                const half zeroFact,       // i13
                                                const uint32_t runSz,      // i12
                                                const uint32_t strideC) {  // i11

    __asm volatile(
            "nop  \n"
            "cmu.cp.128.16.r v0, i14.0 \n"
            "cmu.cp.128.16.r v1, i13.0 \n"
            "lsu0.ldi.128   v2, i17 || iau.incs i12, -8 \n"
            "lsu1.ldi.128   v3, i16 || peu.pcix LTE 0 || bru.bra pqNchwNhwcC3_endloop0\n"
            "lsu0.ldi.128   v4, i15 \n"
            "nop 5 \n"
            "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
            "vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 \n"
            "vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15  \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1     || iau.incs i12, -8\n"
            "CMU.CP.64.F16.U8S V6.0, V10 || peu.pcix LTE 0 || bru.bra pqNchwNhwcC3_endloop1 \n"
            "CMU.CP.64.F16.U8S V7.0, V11 \n"
            "CMU.CP.64.F16.U8S V8.0, V12 \n"
            "nop  \n"
            "cmu.vilv.x8 v10, v12, v7, v6 \n"
            "cmu.vilv.x8 v11, v12, v9, v8 \n"
            "cmu.vilv.x16 v14, v15, v11, v10 \n"
            "pqNchwNhwcC3_loop: \n"
            "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
            "vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 \n"
            "vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15  \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1         || LSU0.ST.32 v14.0, i18 || iau.add i18, i18, i11 \n"
            "vau.add.f16 v12, v4, v1         || iau.incs i12, -8 \n"
            "CMU.CP.64.F16.U8S V6.0, V10     || LSU0.ST.32 v14.1, i18 || iau.add i18, i18, i11 || peu.pcix GT 0 || "
            "bru.bra pqNchwNhwcC3_loop \n"
            "CMU.CP.64.F16.U8S V7.0, V11     || LSU0.ST.32 v14.2, i18 || iau.add i18, i18, i11 \n"
            "CMU.CP.64.F16.U8S V8.0, V12     || LSU0.ST.32 v14.3, i18 || iau.add i18, i18, i11 \n"
            "                                   LSU0.ST.32 v15.0, i18 || iau.add i18, i18, i11 \n"
            "cmu.vilv.x8 v10, v12, v7, v6    || LSU0.ST.32 v15.1, i18 || iau.add i18, i18, i11 \n"
            "cmu.vilv.x8 v11, v12, v9, v8    || LSU0.ST.32 v15.2, i18 || iau.add i18, i18, i11 \n"
            "cmu.vilv.x16 v14, v15, v11, v10 || LSU0.ST.32 v15.3, i18 || iau.add i18, i18, i11 \n"
            "pqNchwNhwcC3_endloop1: \n"
            "LSU0.ST.32 v14.0, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.1, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.2, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.3, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.0, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.1, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.2, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.3, i18 || iau.add i18, i18, i11 \n"
            "pqNchwNhwcC3_endloop0: \n"
            "vau.mul.f16  v2, v2, v0 \n"
            "vau.mul.f16  v3, v3, v0  \n"
            "vau.mul.f16  v4, v4, v0  \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1 \n"
            "CMU.CP.64.F16.U8S V6.0, V10 \n"
            "CMU.CP.64.F16.U8S V7.0, V11 \n"
            "CMU.CP.64.F16.U8S V8.0, V12 \n"
            "nop  \n"
            "cmu.vilv.x8 v10, v12, v7, v6 \n"
            "cmu.vilv.x8 v11, v12, v9, v8 \n"
            "cmu.vilv.x16 v14, v15, v11, v10 \n"
            "LSU0.ST.32 v14.0, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.1, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.2, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v14.3, i18 || iau.add i18, i18, i11 || BRU.jmp i30 \n"
            "LSU0.ST.32 v15.0, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.1, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.2, i18 || iau.add i18, i18, i11 \n"
            "LSU0.ST.32 v15.3, i18 || iau.add i18, i18, i11 \n"
            "nop 2 \n"
            :
            : "r"(out), "r"(in0), "r"(in1), "r"(in2), "r"(scaleFact), "r"(zeroFact), "r"(runSz), "r"(strideC)
            : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC1_asm(uchar8* __restrict out,    // i18
                                                half8* __restrict in0,     // i17
                                                const half scaleFact,      // i16
                                                const half zeroFact,       // i15
                                                const uint32_t runSz,      // i14
                                                const uint32_t strideC) {  // i13
    __asm volatile(
            "nop  \n"
            "cmu.cp.128.16.r v0, i16.0 || LSU0.LDIL i0, 64\n"
            "cmu.cp.128.16.r v1, i15.0 || IAU.ADD i1, i17, 16 \n"
            "lsu0.ldi.128   v2, i17, i0                                        || iau.incs i14, -32 \n"
            "lsu1.ldi.128   v3,  i1, i0 || IAU.ADD i2, i1, 16 || peu.pcix LTE 0 || bru.bra pqNchwNhwcC1_endloop0 \n"
            "lsu0.ldi.128   v4,  i2, i0 || IAU.ADD i3, i2, 16\n"
            "lsu1.ldi.128   v5,  i3, i0 \n"
            "nop 4 \n"
            "vau.mul.f16  v2, v2, v0  || lsu0.ldi.128   v2, i17, i0 || iau.incs i14, -32 \n"
            "vau.mul.f16  v3, v3, v0  || lsu1.ldi.128   v3,  i1, i0 || peu.pcix LTE 0 || bru.bra "
            "pqNchwNhwcC1_endloop1\n"
            "vau.mul.f16  v4, v4, v0  || lsu0.ldi.128   v4,  i2, i0 \n"
            "vau.mul.f16  v5, v5, v0  || lsu1.ldi.128   v5,  i3, i0 \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1  \n"
            "vau.add.f16 v13, v5, v1  \n"
            "pqNchwNhwcC1_loop: \n"
            "vau.mul.f16  v2, v2, v0  || lsu0.ldi.128   v2, i17, i0 || iau.incs i14, -32 \n"
            "vau.mul.f16  v3, v3, v0  || lsu1.ldi.128   v3,  i1, i0 || peu.pcix GT 0 || bru.bra pqNchwNhwcC1_loop\n"
            "vau.mul.f16  v4, v4, v0  || lsu0.ldi.128   v4,  i2, i0 \n"
            "vau.mul.f16  v5, v5, v0  || lsu1.ldi.128   v5,  i3, i0 \n"
            "vau.add.f16 v10, v2, v1  || LSU0.ST.64.F16.U8S V10, I18 || iau.incs i18, 0x8 \n"
            "vau.add.f16 v11, v3, v1  || LSU0.ST.64.F16.U8S V11, I18 || iau.incs i18, 0x8 \n"
            "vau.add.f16 v12, v4, v1  || LSU0.ST.64.F16.U8S V12, I18 || iau.incs i18, 0x8 \n"
            "vau.add.f16 v13, v5, v1  || LSU0.ST.64.F16.U8S V13, I18 || iau.incs i18, 0x8 \n"
            "pqNchwNhwcC1_endloop1: \n"
            "LSU0.ST.64.F16.U8S V10, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V11, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V12, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V13, I18 || iau.incs i18, 0x8 \n"
            "pqNchwNhwcC1_endloop0: \n"
            "vau.mul.f16  v2, v2, v0  \n"
            "vau.mul.f16  v3, v3, v0  \n"
            "vau.mul.f16  v4, v4, v0  \n"
            "vau.mul.f16  v5, v5, v0  \n"
            "vau.add.f16 v10, v2, v1  \n"
            "vau.add.f16 v11, v3, v1  \n"
            "vau.add.f16 v12, v4, v1  \n"
            "vau.add.f16 v13, v5, v1  || BRU.jmp i30 \n"
            "LSU0.ST.64.F16.U8S V10, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V11, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V12, I18 || iau.incs i18, 0x8 \n"
            "LSU0.ST.64.F16.U8S V13, I18 || iau.incs i18, 0x8 \n"
            "nop 6 \n"
            :
            : "r"(out), "r"(in0), "r"(scaleFact), "r"(zeroFact), "r"(runSz), "r"(strideC)
            : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC1_asm_stride(uchar8* __restrict out,    // i18
                                                       half8* __restrict in0,     // i17
                                                       const half scaleFact,      // i16
                                                       const half zeroFact,       // i15
                                                       const uint32_t runSz,      // i14
                                                       const uint32_t strideC) {  // i13
    __asm volatile("nop  \n"
                   "iau.mul i8, i13, 0x10 \n"
                   "CMU.CP.32 I5, I18 \n"
                   "cmu.cp.128.16.r v0, i16.0 || LSU0.LDIL i0, 64\n"
                   "cmu.cp.128.16.r v1, i15.0 || IAU.ADD i1, i17, 16 \n"
                   "lsu0.ldi.128   v2, i17, i0                                        || iau.incs i14, -32 \n"
                   "lsu1.ldi.128   v3,  i1, i0 || IAU.ADD i2, i1, 16 || peu.pcix LTE 0 || bru.bra "
                   "pqNchwNhwcC1AsmStride_endloop0 \n"
                   "lsu0.ldi.128   v4,  i2, i0 || IAU.ADD i3, i2, 16\n"
                   "lsu1.ldi.128   v5,  i3, i0 \n"
                   "nop 4 \n"
                   "vau.mul.f16  v2, v2, v0  || lsu0.ldi.128   v2, i17, i0 || iau.incs i14, -32 \n"
                   "vau.mul.f16  v3, v3, v0  || lsu1.ldi.128   v3,  i1, i0 || peu.pcix LTE 0 || bru.bra "
                   "pqNchwNhwcC1AsmStride_endloop1\n"
                   "vau.mul.f16  v4, v4, v0  || lsu0.ldi.128   v4,  i2, i0 \n"
                   "vau.mul.f16  v5, v5, v0  || lsu1.ldi.128   v5,  i3, i0 \n"
                   "vau.add.f16 v10, v2, v1  \n"
                   "vau.add.f16 v11, v3, v1  \n"
                   "vau.add.f16 v12, v4, v1  \n"
                   "vau.add.f16 v13, v5, v1  \n"

                   "pqNchwNhwcC1AsmStride_loop: \n"
                   "CMU.CP.64.F16.U8S V6.0, V10 || lsu0.ldi.128   v2, i17, i0  \n"
                   "CMU.CP.64.F16.U8S V6.1, V11 || lsu1.ldi.128   v3,  i1, i0  \n"
                   "CMU.CP.64.F16.U8S V7.0, V12 || lsu0.ldi.128   v4,  i2, i0 \n"
                   "CMU.CP.64.F16.U8S V7.1, V13 || lsu1.ldi.128   v5,  i3, i0    || iau.add i4, I5, 0 \n"
                   "vau.mul.f16  v10, v2, v0                                     || iau.add I5, I4, i8   \n"
                   "vau.mul.f16  v11, v3, v0 || LSU0.STi.8  V6.0, I4, i13 || LSU1.STi.8  V7.0, I5, i13   \n"
                   "vau.mul.f16  v12, v4, v0 || LSU0.STi.8  V6.1, I4, i13 || LSU1.STi.8  V7.1, I5, i13   \n"
                   "vau.mul.f16  v13, v5, v0 || LSU0.STi.8  V6.2, I4, i13 || LSU1.STi.8  V7.2, I5, i13   \n"
                   "vau.add.f16 v10, v10, v1 || LSU0.STi.8  V6.3, I4, i13 || LSU1.STi.8  V7.3, I5, i13   \n"
                   "vau.add.f16 v11, v11, v1 || LSU0.STi.8  V6.4, I4, i13 || LSU1.STi.8  V7.4, I5, i13   \n"
                   "vau.add.f16 v12, v12, v1 || LSU0.STi.8  V6.5, I4, i13 || LSU1.STi.8  V7.5, I5, i13   \n"
                   "vau.add.f16 v13, v13, v1 || LSU0.STi.8  V6.6, I4, i13 || LSU1.STi.8  V7.6, I5, i13   \n"
                   "LSU0.STi.8  V6.7, I4, i13 || LSU1.STi.8  V7.7, I5, i13 \n"
                   "LSU0.STi.8  V6.8, I4, i13 || LSU1.STi.8  V7.8, I5, i13 \n"
                   "LSU0.STi.8  V6.9, I4, i13 || LSU1.STi.8  V7.9, I5, i13 \n"
                   "LSU0.STi.8 V6.10, I4, i13 || LSU1.STi.8 V7.10, I5, i13 \n"
                   "LSU0.STi.8 V6.11, I4, i13 || LSU1.STi.8 V7.11, I5, i13 || iau.incs i14, -32 \n"
                   "LSU0.STi.8 V6.12, I4, i13 || LSU1.STi.8 V7.12, I5, i13 || peu.pcix GT 0 || bru.bra "
                   "pqNchwNhwcC1AsmStride_loop \n"
                   "LSU0.STi.8 V6.13, I4, i13 || LSU1.STi.8 V7.13, I5, i13 \n"
                   "LSU0.STi.8 V6.14, I4, i13 || LSU1.STi.8 V7.14, I5, i13 \n"
                   "LSU0.STi.8 V6.15, I4, i13 || LSU1.STi.8 V7.15, I5, i13 \n"
                   "pqNchwNhwcC1AsmStride_endloop1: \n"
                   "CMU.CP.64.F16.U8S V6.0, V10 \n"
                   "CMU.CP.64.F16.U8S V6.1, V11 \n"
                   "CMU.CP.64.F16.U8S V7.0, V12 || iau.add i4, I5, 0 \n"
                   "CMU.CP.64.F16.U8S V7.1, V13 || iau.add I5, I4, i8 \n"
                   "nop  \n"
                   "LSU0.STi.8  V6.0, I4, i13 || LSU1.STi.8  V7.0, I5, i13 \n"
                   "LSU0.STi.8  V6.1, I4, i13 || LSU1.STi.8  V7.1, I5, i13 \n"
                   "LSU0.STi.8  V6.2, I4, i13 || LSU1.STi.8  V7.2, I5, i13 \n"
                   "LSU0.STi.8  V6.3, I4, i13 || LSU1.STi.8  V7.3, I5, i13 \n"
                   "LSU0.STi.8  V6.4, I4, i13 || LSU1.STi.8  V7.4, I5, i13 \n"
                   "LSU0.STi.8  V6.5, I4, i13 || LSU1.STi.8  V7.5, I5, i13 \n"
                   "LSU0.STi.8  V6.6, I4, i13 || LSU1.STi.8  V7.6, I5, i13 \n"
                   "LSU0.STi.8  V6.7, I4, i13 || LSU1.STi.8  V7.7, I5, i13 \n"
                   "LSU0.STi.8  V6.8, I4, i13 || LSU1.STi.8  V7.8, I5, i13 \n"
                   "LSU0.STi.8  V6.9, I4, i13 || LSU1.STi.8  V7.9, I5, i13 \n"
                   "LSU0.STi.8 V6.10, I4, i13 || LSU1.STi.8 V7.10, I5, i13 \n"
                   "LSU0.STi.8 V6.11, I4, i13 || LSU1.STi.8 V7.11, I5, i13 \n"
                   "LSU0.STi.8 V6.12, I4, i13 || LSU1.STi.8 V7.12, I5, i13 \n"
                   "LSU0.STi.8 V6.13, I4, i13 || LSU1.STi.8 V7.13, I5, i13 \n"
                   "LSU0.STi.8 V6.14, I4, i13 || LSU1.STi.8 V7.14, I5, i13 \n"
                   "LSU0.STi.8 V6.15, I4, i13 || LSU1.STi.8 V7.15, I5, i13 \n"
                   "pqNchwNhwcC1AsmStride_endloop0: \n"
                   "vau.mul.f16  v2, v2, v0  \n"
                   "vau.mul.f16  v3, v3, v0 || iau.add i4, I5, 0 \n"
                   "vau.mul.f16  v4, v4, v0 || iau.add I5, I4, i8 \n"
                   "vau.mul.f16  v5, v5, v0  \n"
                   "vau.add.f16 v10, v2, v1  \n"
                   "vau.add.f16 v11, v3, v1  \n"
                   "vau.add.f16 v12, v4, v1  \n"
                   "vau.add.f16 v13, v5, v1   \n"
                   "CMU.CP.64.F16.U8S V6.0, V10 \n"
                   "CMU.CP.64.F16.U8S V6.1, V11 \n"
                   "CMU.CP.64.F16.U8S V7.0, V12 \n"
                   "CMU.CP.64.F16.U8S V7.1, V13 \n"
                   "nop  \n"
                   "LSU0.STi.8  V6.0, I4, i13 || LSU1.STi.8  V7.0, I5, i13 \n"
                   "LSU0.STi.8  V6.1, I4, i13 || LSU1.STi.8  V7.1, I5, i13 \n"
                   "LSU0.STi.8  V6.2, I4, i13 || LSU1.STi.8  V7.2, I5, i13 \n"
                   "LSU0.STi.8  V6.3, I4, i13 || LSU1.STi.8  V7.3, I5, i13 \n"
                   "LSU0.STi.8  V6.4, I4, i13 || LSU1.STi.8  V7.4, I5, i13 \n"
                   "LSU0.STi.8  V6.5, I4, i13 || LSU1.STi.8  V7.5, I5, i13 \n"
                   "LSU0.STi.8  V6.6, I4, i13 || LSU1.STi.8  V7.6, I5, i13 \n"
                   "LSU0.STi.8  V6.7, I4, i13 || LSU1.STi.8  V7.7, I5, i13 \n"
                   "LSU0.STi.8  V6.8, I4, i13 || LSU1.STi.8  V7.8, I5, i13 \n"
                   "LSU0.STi.8  V6.9, I4, i13 || LSU1.STi.8  V7.9, I5, i13 \n"
                   "LSU0.STi.8 V6.10, I4, i13 || LSU1.STi.8 V7.10, I5, i13 || BRU.jmp i30 \n"
                   "LSU0.STi.8 V6.11, I4, i13 || LSU1.STi.8 V7.11, I5, i13 \n"
                   "LSU0.STi.8 V6.12, I4, i13 || LSU1.STi.8 V7.12, I5, i13 \n"
                   "LSU0.STi.8 V6.13, I4, i13 || LSU1.STi.8 V7.13, I5, i13 \n"
                   "LSU0.STi.8 V6.14, I4, i13 || LSU1.STi.8 V7.14, I5, i13 \n"
                   "LSU0.STi.8 V6.15, I4, i13 || LSU1.STi.8 V7.15, I5, i13 \n"
                   "nop  \n"
                   "nop  \n"
                   :
                   : "r"(out), "r"(in0), "r"(scaleFact), "r"(zeroFact), "r"(runSz), "r"(strideC)
                   : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC3Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          half8* __restrict in1,   // i16
                                                          half8* __restrict in2,   // i15
                                                          const half scaleFact,    // i14
                                                          const half zeroFact,     // i13
                                                          const uint32_t runSz) {  // i12

    __asm volatile("lsu0.ldi.128   v2, i17 \n"
                   "lsu1.ldi.128   v3, i16 \n"
                   "lsu0.ldi.128   v4, i15  \n"
                   "lsu1.ldi.128   v5, i17 \n"
                   "lsu0.ldi.128   v6, i16 \n"
                   "lsu1.ldi.128   v7, i15 \n"
                   "cmu.cp.128.16.r v0, i14.0 \n"
                   "cmu.cp.128.16.r v1, i13.0 \n"

                   "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
                   "vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 \n"
                   "vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 \n"
                   "vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i17 \n"
                   "vau.mul.f16  v6, v6, v0 || lsu0.ldi.128   v6, i16 \n"
                   "vau.mul.f16  v7, v7, v0 || lsu1.ldi.128   v7, i15 \n"
                   "vau.add.f16 v10, v2, v1  \n"
                   "vau.add.f16 v11, v3, v1  \n"
                   "vau.add.f16 v12, v4, v1 || CMU.CP.32 I1, I18 \n"
                   "vau.add.f16 v13, v5, v1 || iau.add I2, I1, 8 \n"
                   "vau.add.f16 v14, v6, v1 || lsu0.ldil i8, 16 \n"
                   "vau.add.f16 v15, v7, v1 || iau.incs i12, -32 \n"

                   "pqNchwNhwcC3Exp4Algn16_asm_loop: \n"
                   "cmu.vilv.x16 v16, v17, v11, v10 || vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
                   "cmu.vilv.x16 v18, v19, v19, v12 || vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 \n"
                   "cmu.vilv.x16 v20, v21, v14, v13 || vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 \n"
                   "cmu.vilv.x16 v22, v23, v23, v15 || vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i17 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16 || vau.mul.f16  v6, v6, v0 || lsu0.ldi.128   v6, i16 || iau.incs "
                   "i12, -16 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17 || vau.mul.f16  v7, v7, v0 || lsu1.ldi.128   v7, i15 || peu.pcix "
                   "GT 0 ||  bru.bra pqNchwNhwcC3Exp4Algn16_asm_loop \n"
                   "cmu.vilv.x32 v20, v22, v22, v20 || vau.add.f16 v10, v2, v1  \n"
                   "cmu.vilv.x32 v21, v23, v23, v21 || vau.add.f16 v11, v3, v1  \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 || vau.add.f16 v12, v4, v1 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8 || vau.add.f16 v13, v5, v1  \n"
                   "LSU0.STI.64.F16.U8S V20, I1, i8 || LSU1.STI.64.F16.U8S V22, I2, i8 || vau.add.f16 v14, v6, v1  \n"
                   "LSU0.STI.64.F16.U8S V21, I1, i8 || LSU1.STI.64.F16.U8S V23, I2, i8 || vau.add.f16 v15, v7, v1 \n"

                   "cmu.vilv.x16 v16, v17, v11, v10 || vau.mul.f16  v2, v2, v0 \n"
                   "cmu.vilv.x16 v18, v19, v19, v12 || vau.mul.f16  v3, v3, v0 \n"
                   "cmu.vilv.x16 v20, v21, v14, v13 || vau.mul.f16  v4, v4, v0 \n"
                   "cmu.vilv.x16 v22, v23, v23, v15 || vau.mul.f16  v5, v5, v0 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16 || vau.mul.f16  v6, v6, v0 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17 || vau.mul.f16  v7, v7, v0 \n"
                   "cmu.vilv.x32 v20, v22, v22, v20 || vau.add.f16 v10, v2, v1  \n"
                   "cmu.vilv.x32 v21, v23, v23, v21 || vau.add.f16 v11, v3, v1  \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 || vau.add.f16 v12, v4, v1 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8 || vau.add.f16 v13, v5, v1  \n"
                   "LSU0.STI.64.F16.U8S V20, I1, i8 || LSU1.STI.64.F16.U8S V22, I2, i8 || vau.add.f16 v14, v6, v1  \n"
                   "LSU0.STI.64.F16.U8S V21, I1, i8 || LSU1.STI.64.F16.U8S V23, I2, i8 || vau.add.f16 v15, v7, v1 \n"

                   "cmu.vilv.x16 v16, v17, v11, v10 \n"
                   "cmu.vilv.x16 v18, v19, v19, v12 \n"
                   "cmu.vilv.x16 v20, v21, v14, v13 \n"
                   "cmu.vilv.x16 v22, v23, v23, v15 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17 \n"
                   "cmu.vilv.x32 v20, v22, v22, v20 || BRU.jmp i30 \n"
                   "cmu.vilv.x32 v21, v23, v23, v21  \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8  \n"
                   "LSU0.STI.64.F16.U8S V20, I1, i8 || LSU1.STI.64.F16.U8S V22, I2, i8  \n"
                   "LSU0.STI.64.F16.U8S V21, I1, i8 || LSU1.STI.64.F16.U8S V23, I2, i8 \n"
                   "nop\n"
                   "nop\n"
                   :
                   : "r"(out), "r"(in0), "r"(in1), "r"(in2), "r"(scaleFact), "r"(zeroFact), "r"(runSz)
                   : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC4Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          half8* __restrict in1,   // i16
                                                          half8* __restrict in2,   // i15
                                                          half8* __restrict in3,   // i14
                                                          const half scaleFact,    // i13
                                                          const half zeroFact,     // i12
                                                          const uint32_t runSz) {  // i11

    __asm volatile("lsu0.ldi.128   v2, i17 \n"
                   "lsu1.ldi.128   v3, i16 \n"
                   "lsu0.ldi.128   v4, i15 \n"
                   "lsu1.ldi.128   v5, i14 || cmu.cp.128.16.r v0, i13.0\n"
                   "cmu.cp.128.16.r v1, i12.0 \n"
                   "CMU.CP.32 I1, I18 \n"
                   "iau.add I2, I1, 8 \n"
                   "lsu0.ldil i8, 16 \n"
                   "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 \n"
                   "vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 \n"
                   "vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 \n"
                   "vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i14 \n"
                   "vau.add.f16 v10, v2, v1 \n"
                   "vau.add.f16 v11, v3, v1 \n"
                   "vau.add.f16 v12, v4, v1 \n"
                   "vau.add.f16 v13, v5, v1 \n"

                   "pqNchwNhwcC4Exp4Algn16_asm_loop: \n"
                   "vau.mul.f16  v2, v2, v0 || lsu0.ldi.128   v2, i17 || iau.incs i11, -8 \n"
                   "cmu.vilv.x16 v16, v17, v11, v10||vau.mul.f16  v3, v3, v0 || lsu1.ldi.128   v3, i16 || peu.pcix GT "
                   "0 ||  bru.bra pqNchwNhwcC4Exp4Algn16_asm_loop\n"
                   "cmu.vilv.x16 v18, v19, v13, v12||vau.mul.f16  v4, v4, v0 || lsu0.ldi.128   v4, i15 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16||vau.mul.f16  v5, v5, v0 || lsu1.ldi.128   v5, i14 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17||vau.add.f16 v10, v2, v1 \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 || vau.add.f16 v11, v3, v1 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8 || vau.add.f16 v12, v4, v1 \n"
                   "vau.add.f16 v13, v5, v1 \n"

                   "vau.mul.f16  v2, v2, v0 \n"
                   "cmu.vilv.x16 v16, v17, v11, v10 || vau.mul.f16  v3, v3, v0 \n"
                   "cmu.vilv.x16 v18, v19, v13, v12 || vau.mul.f16  v4, v4, v0 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16 || vau.mul.f16  v5, v5, v0 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17 || vau.add.f16 v10, v2, v1 \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 || vau.add.f16 v11, v3, v1 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8 || vau.add.f16 v12, v4, v1 \n"
                   "vau.add.f16 v13, v5, v1 \n"
                   "nop \n"
                   "cmu.vilv.x16 v16, v17, v11, v10 || BRU.jmp i30 \n"
                   "cmu.vilv.x16 v18, v19, v13, v12 \n"
                   "cmu.vilv.x32 v16, v18, v18, v16 \n"
                   "cmu.vilv.x32 v17, v19, v19, v17 \n"
                   "LSU0.STI.64.F16.U8S V16, I1, i8 || LSU1.STI.64.F16.U8S V18, I2, i8 \n"
                   "LSU0.STI.64.F16.U8S V17, I1, i8 || LSU1.STI.64.F16.U8S V19, I2, i8  \n"
                   "nop \n"
                   "nop \n"
                   :
                   : "r"(out), "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(scaleFact), "r"(zeroFact), "r"(runSz)
                   : "memory");
}

void __attribute__((noinline)) pqNchwNhwcC1Exp4Algn16_asm(uchar8* __restrict out,  // i18
                                                          half8* __restrict in0,   // i17
                                                          const half scaleFact,    // i16
                                                          const half zeroFact,     // i15
                                                          const uint32_t runSz) {  // i14

    __asm volatile("LSU0.LDO.64 V2.0 , i17, 0x00 || LSU1.LDO.64 V2.1 , i17, 0x08 \n"
                   "LSU0.LDO.64 V3.0 , i17, 0x10 || LSU1.LDO.64 V3.1 , i17, 0x18 || iau.incs i17, 0x20\n"
                   "cmu.cp.128.16.r v0, i16.0 \n"
                   "cmu.cp.128.16.r v1, i15.0 \n"
                   "lsu0.ldil i0, 255 \n"
                   "CMU.CP.128.I16.F16.R V22, I0.0 \n"
                   "nop \n"
                   "vau.mul.f16  v2, v2, v0 || LSU0.LDO.64 V2.0 , i17, 0x00 || LSU1.LDO.64 V2.1 , i17, 0x08 \n"
                   "vau.mul.f16  v3, v3, v0 || LSU0.LDO.64 V3.0 , i17, 0x10 || LSU1.LDO.64 V3.1 , i17, 0x18 || "
                   "iau.incs i17, 0x20 \n"
                   "nop \n"
                   "vau.add.f16 v8, v2, v1 \n"
                   "vau.add.f16 v9, v3, v1 \n"
                   "nop \n"
                   "nop \n"
                   "vau.mul.f16  v2, v2, v0 || LSU0.LDO.64 V2.0 , i17, 0x00 || LSU1.LDO.64 V2.1 , i17, 0x08 \n"
                   "cmu.clamp0.f16 v10, v8, v22    || vau.mul.f16  v3, v3, v0 || LSU0.LDO.64 V3.0 , i17, 0x10 || "
                   "LSU1.LDO.64 V3.1 , i17, 0x18 || iau.incs i17, 0x20 \n"
                   "cmu.clamp0.f16 v11, v9, v22     \n"
                   "cmu.cp.128.f16.u32s v12, v10.0 || vau.add.f16 v8, v2, v1 \n"
                   "cmu.cp.128.f16.u32s v13, v10.1 || vau.add.f16 v9, v3, v1 \n"
                   "cmu.cp.128.f16.u32s v14, v11.0   \n"
                   "cmu.cp.128.f16.u32s v15, v11.1 || iau.incs i14, -64 \n"

                   "pqNchwNhwcC1Exp4Algn16_asm_loop: \n"
                   "vau.mul.f16  v2, v2, v0 || LSU0.LDO.64 V2.0 , i17, 0x00 || LSU1.LDO.64 V2.1 , i17, 0x08 || "
                   "peu.pcix GT 0 ||  bru.bra pqNchwNhwcC1Exp4Algn16_asm_loop \n"
                   "cmu.clamp0.f16 v10, v8, v22    || vau.mul.f16  v3, v3, v0 || LSU0.LDO.64 V3.0 , i17, 0x10 || "
                   "LSU1.LDO.64 V3.1 , i17, 0x18 || iau.incs i17, 0x20 \n"
                   "LSU0.STO.64 V12.0 , i18, 0x00 || LSU1.STO.64 V12.1 , i18, 0x08 || cmu.clamp0.f16 v11, v9, v22   \n"
                   "LSU0.STO.64 V13.0 , i18, 0x10 || LSU1.STO.64 V13.1 , i18, 0x18 || cmu.cp.128.f16.u32s v12, v10.0 "
                   "|| vau.add.f16 v8, v2, v1 \n"
                   "LSU0.STO.64 V14.0 , i18, 0x20 || LSU1.STO.64 V14.1 , i18, 0x28 || cmu.cp.128.f16.u32s v13, v10.1 "
                   "|| vau.add.f16 v9, v3, v1 \n"
                   "LSU0.STO.64 V15.0 , i18, 0x30 || LSU1.STO.64 V15.1 , i18, 0x38 || iau.incs i18, 0x40 || "
                   "cmu.cp.128.f16.u32s v14, v11.0 \n"
                   "cmu.cp.128.f16.u32s v15, v11.1 || iau.incs i14, -16 \n"

                   "vau.mul.f16  v2, v2, v0 \n"
                   "cmu.clamp0.f16 v10, v8, v22    || vau.mul.f16  v3, v3, v0 \n"
                   "LSU0.STO.64 V12.0 , i18, 0x00 || LSU1.STO.64 V12.1 , i18, 0x08 || cmu.clamp0.f16 v11, v9, v22   \n"
                   "LSU0.STO.64 V13.0 , i18, 0x10 || LSU1.STO.64 V13.1 , i18, 0x18 || cmu.cp.128.f16.u32s v12, v10.0 "
                   "|| vau.add.f16 v8, v2, v1 \n"
                   "LSU0.STO.64 V14.0 , i18, 0x20 || LSU1.STO.64 V14.1 , i18, 0x28 || cmu.cp.128.f16.u32s v13, v10.1 "
                   "|| vau.add.f16 v9, v3, v1 \n"
                   "LSU0.STO.64 V15.0 , i18, 0x30 || LSU1.STO.64 V15.1 , i18, 0x38 || iau.incs i18, 0x40 || "
                   "cmu.cp.128.f16.u32s v14, v11.0 \n"
                   "cmu.cp.128.f16.u32s v15, v11.1  \n"
                   "cmu.clamp0.f16 v10, v8, v22    \n"
                   "LSU0.STO.64 V12.0 , i18, 0x00 || LSU1.STO.64 V12.1 , i18, 0x08 || cmu.clamp0.f16 v11, v9, v22   \n"
                   "LSU0.STO.64 V13.0 , i18, 0x10 || LSU1.STO.64 V13.1 , i18, 0x18 || cmu.cp.128.f16.u32s v12, v10.0 \n"
                   "LSU0.STO.64 V14.0 , i18, 0x20 || LSU1.STO.64 V14.1 , i18, 0x28 || cmu.cp.128.f16.u32s v13, v10.1 "
                   "|| BRU.jmp i30\n"
                   "LSU0.STO.64 V15.0 , i18, 0x30 || LSU1.STO.64 V15.1 , i18, 0x38 || iau.incs i18, 0x40 || "
                   "cmu.cp.128.f16.u32s v14, v11.0 \n"
                   "cmu.cp.128.f16.u32s v15, v11.1 || LSU0.STO.64 V12.0 , i18, 0x00 || LSU1.STO.64 V12.1 , i18, 0x08 \n"
                   "LSU0.STO.64 V13.0 , i18, 0x10 || LSU1.STO.64 V13.1 , i18, 0x18 \n"
                   "LSU0.STO.64 V14.0 , i18, 0x20 || LSU1.STO.64 V14.1 , i18, 0x28 \n"
                   "LSU0.STO.64 V15.0 , i18, 0x30 || LSU1.STO.64 V15.1 , i18, 0x38 \n"
                   "nop\n"
                   "nop\n"
                   :
                   : "r"(out), "r"(in0), "r"(scaleFact), "r"(zeroFact), "r"(runSz)
                   : "memory");
}
