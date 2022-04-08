//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorConvert.h>
#include "mvSubspaces.h"

#include <math.h>
#include <moviVectorTypes.h>
#include <param_mvn.h>

#define NCHW_REV 0x4321  // reverse code of ND_NCHW 0x1234
#define NCWH_REV 0x3421  // reverse code of ND_NCWH 0x1243
#define NHWC_REV 0x2431  // reverse code of ND_NHWC 0x1342
#define NWHC_REV 0x2341  // reverse code of ND_NWHC 0x1432

#define VEC_SZ 8  // SIMD vector size

using namespace sw_params;
using namespace subspace;

namespace {

struct MvnArgs {
    uint32_t chMul;  // channel multiplier (for C < VEC_SZ)
    uint32_t step;   // increment for vector access
    uint32_t nElem;
    uint32_t C;  // current number of channels
    float div;   // mean normalization factor
    float eps;
    float8 mean;
    float8 norm;  //'1.0f/sqrtf(var[ch]+eps)' or '1.0'
    bool merge;   // across=true or NCHW layout
};

void mvn_calc_norm_0(half* __restrict in, MvnArgs& p);
void mvn_calc_norm_1(half* __restrict in, MvnArgs& p);

void mvn_apply_const(half* __restrict in, half* __restrict out, MvnArgs& args);
void mvn_apply_NHWC_small(half* __restrict in, half* __restrict out, MvnArgs& args);
void mvn_apply_NHWC_tail(half* in, half* out, MvnArgs& args);

void mvn_NCHW(const MvnParams* p) {
    int32_t* dims = (int32_t*)(p->input.dimsAddr);
    half* in = (half*)(p->input.dataAddr);
    half* out = (half*)(p->output.dataAddr);

    uint32_t W = dims[0];
    uint32_t H = dims[1];
    uint32_t C = dims[2];
    uint32_t N = dims[3];

    MvnArgs arg;
    arg.eps = p->eps;
    arg.C = 1;
    arg.chMul = 1;
    arg.step = VEC_SZ;
    arg.merge = true;

    const auto pfn_calc = p->normalize ? mvn_calc_norm_1 : mvn_calc_norm_0;

    for (uint32_t n = 0; n < N; n++) {
        if (!p->acrossChannels) {  // calc & update each ch independently
            arg.div = 1.0 / (W * H);
            arg.nElem = H * W;
            for (uint32_t ch = 0; ch < C; ch++) {
                pfn_calc(in + ch * H * W, arg);
                arg.mean = arg.mean[0];
                arg.norm = arg.norm[0];
                mvn_apply_const(in + ch * H * W, out + ch * H * W, arg);
            }
        } else {  // calc & update all channels
            arg.div = 1.0 / (C * W * H);
            arg.nElem = C * H * W;
            pfn_calc(in, arg);
            arg.mean = arg.mean[0];
            arg.norm = arg.norm[0];
            mvn_apply_const(in, out, arg);
        }

        in += C * H * W;
        out += C * H * W;
    }
}

void mvn_NHWC(const MvnParams* param) {
    int32_t* dims = (int32_t*)(param->input.dimsAddr);
    half* in = (half*)(param->input.dataAddr);
    half* out = (half*)(param->output.dataAddr);

    uint32_t C = dims[0];
    uint32_t W = dims[1];
    uint32_t H = dims[2];
    uint32_t N = dims[3];

    const auto pfn_calc = param->normalize ? mvn_calc_norm_1 : mvn_calc_norm_0;

    MvnArgs p;
    p.eps = param->eps;
    p.div = 1.0 / (W * H);
    p.nElem = H * W * C;
    p.merge = false;

    for (uint32_t n = 0; n < N; n++) {
        if (C < VEC_SZ) {
            p.chMul = VEC_SZ / C;
            p.step = p.chMul * C;
            p.C = C;
            pfn_calc(in, p);
            mvn_apply_NHWC_small(in, out, p);
        } else {  // C >= VEC_SZ
            uint32_t ch;
            p.chMul = 1;
            p.step = C;
            p.C = VEC_SZ;
            // compute groups of nearby 8xchannels
            for (ch = 0; ch < (C / VEC_SZ) * VEC_SZ; ch += VEC_SZ) {
                pfn_calc(in + ch, p);
                mvn_apply_const(in + ch, out + ch, p);
            }
            // trailing channels
            uint32_t remCh = C - ch;
            if (remCh) {
                pfn_calc(in + ch, p);
                p.C = remCh;
                mvn_apply_NHWC_tail(in + ch, out + ch, p);
            }
        }

        in += C * H * W;
        out += C * H * W;
    }
}

// returns {mean, norm = 1.0}
void mvn_calc_norm_0(half* __restrict in, MvnArgs& p) {
    uint32_t nVec = p.nElem / p.step;
    uint32_t nScl = p.nElem - nVec * p.step;
    float8 sum1 = 0;  // sum(xi)

    // vector loop
    half* __restrict pin = in;
#pragma clang loop unroll_count(8)
    for (uint32_t i = 0; i < nVec; i++) {
        float8 x = mvuConvert_float8(*((half8*)pin));
        sum1 += x;
        pin += p.step;
    }

    // consolidate vector loop results
    for (int j = 1; j < p.chMul; j++) {
        for (int ch = 0; ch < p.C; ch++) {
            sum1[ch] += sum1[j * p.C + ch];
        }
    }

    // trailing scalars
    for (uint32_t ch = 0; ch < nScl; ch++) {
        float x = pin[ch];
        sum1[ch] += x;
    }

    if (p.merge) {  // merge vector loop results into single value
        float m = __builtin_shave_sau_sumx_f32_r(sum1.lo) + __builtin_shave_sau_sumx_f32_r(sum1.hi);
        sum1 = m;
    }

    p.mean = sum1 * p.div;
    p.norm = 1.0;
}

// returns {mean, norm = 1.0/variance}
void mvn_calc_norm_1(half* __restrict in, MvnArgs& p) {
    uint32_t nVec = p.nElem / p.step;
    uint32_t nScl = p.nElem - nVec * p.step;
    float8 sum1 = 0;  // sum(xi)
    float8 sum2 = 0;  // sum(xi^2)

    // vector loop
    half* __restrict pin = in;
#pragma clang loop unroll_count(8)
    for (uint32_t i = 0; i < nVec; i++) {
        float8 x = mvuConvert_float8(*((half8*)pin));
        sum1 += x;
        sum2 += x * x;
        pin += p.step;
    }

    // consolidate vector loop results (C-minor, low C case)
    for (int j = 1; j < p.chMul; j++) {
        for (int ch = 0; ch < p.C; ch++) {
            sum1[ch] += sum1[j * p.C + ch];
            sum2[ch] += sum2[j * p.C + ch];
        }
    }

    // trailing scalars
    for (uint32_t ch = 0; ch < nScl; ch++) {
        float x = pin[ch];
        sum1[ch] += x;
        sum2[ch] += x * x;
    }

    if (p.merge) {  // merge vector loop results into single value
        float s1 = __builtin_shave_sau_sumx_f32_r(sum1.lo) + __builtin_shave_sau_sumx_f32_r(sum1.hi);
        float s2 = __builtin_shave_sau_sumx_f32_r(sum2.lo) + __builtin_shave_sau_sumx_f32_r(sum2.hi);
        sum1 = s1;
        sum2 = s2;
    }

    float8 mean = sum1 * p.div;
    float8 sumSq = sum2 * p.div;

    // Compute variance, norm-factor
    float8 norm;
    float8 var = sumSq - mean * mean;
    for (int ch = 0; ch < p.C; ch++) {
        var[ch] = var[ch] < 0.0f ? 1.f : var[ch];
        norm[ch] = 1.0f / sqrtf(var[ch] + p.eps);
    }

    p.mean = mean;
    p.norm = norm;
}

// NHWC, C > VEC_SZ config [apply tail-Channels (<=7) update]
// (not using 'restrict' pointers since doing read-modify-writes)
void mvn_apply_NHWC_tail(half* in, half* out, MvnArgs& p) {
    uint32_t num = p.nElem / p.step;
    half8 vMean = mvuConvert_half8(p.mean);
    half8 vNorm = mvuConvert_half8(p.norm);

    ushort8 mask1 = 0;  // keep mask for existing 'out'
    for (uint32_t ch = p.C; ch < VEC_SZ; ch++) {
        mask1[ch] = 0xFFFF;
    }
    ushort8 mask2 = ~mask1;  // keep mask for tail ch

#pragma clang loop unroll_count(4)
    for (uint32_t i = 0; i < num - 1; i++) {
        half8* vInp = (half8*)in;
        ushort8* uOut = reinterpret_cast<ushort8*>(out);
        half8 tail = (*vInp - vMean) * vNorm;

        // Combine tail ch with existing data (e.g. C=10)
        // uTail = [c8,c9,__,__,__,__,__,__,__,__] (new data)
        // *uOut = [__,__,c0,c1,c2,c3,c4,c5,c6,c7] (computed in prev step)
        ushort8 uTail = *(reinterpret_cast<ushort8*>(&tail));
        ushort8 comb = ((*uOut) & mask1) | (uTail & mask2);
        *uOut = comb;

        in += p.step;
        out += p.step;
    }

    // Final ch in tensor (e.g. C=10)
    // [__,__,__,__,__,__,__,__,c8,c9]
    for (uint32_t ch = 0; ch < p.C; ch++) {
        out[ch] = (in[ch] - vMean[ch]) * vNorm[ch];
    }
}

// out = (in - mean) * norm (mean,norm = 8xConstants)
void mvn_apply_const(half* __restrict in, half* __restrict out, MvnArgs& p) {
    uint32_t nVec = p.nElem / p.step;
    uint32_t nScl = p.nElem - nVec * p.step;

    half8 vMean = mvuConvert_half8(p.mean);
    half8 vNorm = mvuConvert_half8(p.norm);

// vector loop
#pragma clang loop unroll_count(8)
    for (uint32_t i = 0; i < nVec; i++) {
        half8* __restrict vInp = (half8*)in;
        half8* __restrict vOut = (half8*)out;
        *vOut = (*vInp - vMean) * vNorm;
        in += p.step;
        out += p.step;
    }

    // trailing scalars
    for (uint32_t i = 0; i < nScl; i++) {
        out[i] = (in[i] - vMean[i]) * vNorm[i];
    }
}

// NHWC small C (< VEC_SZ)
void mvn_apply_NHWC_small(half* __restrict in, half* __restrict out, MvnArgs& p) {
    // Doing :
    // for(int i=0; i<p.nElem; i++){
    //     int idx =  i % p.C;
    //     out[i] = (in[i] - p.mean[idx]) * p.norm[idx];
    // }

    uint32_t nVec = p.nElem / VEC_SZ;
    uint32_t nScl = p.nElem - nVec * VEC_SZ;

    // Lookup window (replicate mean/norm values for vector access)
    const uint32_t PAD = (VEC_SZ - 1);
    half wMean[p.C + PAD];
    half wNorm[p.C + PAD];
    for (uint32_t i = 0; i < p.C + PAD; i++) {
        wMean[i] = p.mean[i % p.C];
        wNorm[i] = p.norm[i % p.C];
    }

    // vector loop
    half8* __restrict vin = (half8*)in;
    half8* __restrict vout = (half8*)out;
#pragma clang loop unroll_count(8)
    for (uint32_t i = 0; i < nVec; i++) {
        uint32_t idx = (i * VEC_SZ) % p.C;
        half8 m = *(half8*)(wMean + idx);
        half8 n = *(half8*)(wNorm + idx);
        vout[i] = (vin[i] - m) * n;
    }

    // trailing scalars
    if (nScl) {
        uint32_t idx = (nVec * VEC_SZ) % p.C;
        half8 m = *(half8*)(wMean + idx);
        half8 n = *(half8*)(wNorm + idx);
        half8 tail = (vin[nVec] - m) * n;
        // Do only necessary copies
        for (uint32_t i = 0; i < nScl; ++i) {
            out[nVec * VEC_SZ + i] = tail[i];
        }
    }
}

}  // namespace

using namespace subspace;

namespace nn {
namespace shave_lib {

extern "C" {
void singleShaveMVN(uint32_t lParams) {
    const MvnParams* p = reinterpret_cast<const MvnParams*>(lParams);

    // If AcrossCh, doing any layout via faster NCHW impl
    if (p->acrossChannels || p->input.dimsOrder == NCHW_REV || p->input.dimsOrder == NCWH_REV) {
        mvn_NCHW(p);
    } else if (p->input.dimsOrder == NHWC_REV || p->input.dimsOrder == NWHC_REV) {
        mvn_NHWC(p);
    }
}
}

}  // namespace shave_lib
}  // namespace nn
