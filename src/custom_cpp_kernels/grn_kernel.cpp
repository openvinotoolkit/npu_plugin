// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_params.h>

#include <moviVectorTypes.h>
#include <moviVectorUtils.h>
#include <moviVectorFunctions.h>

#include <math.h>
#include <algorithm>
#include <cstdio>

#define FAST_SQRT 1

inline static int32_t divRoundUp(int32_t a, int32_t b) { return (a + b - 1) / b; }

static void GRN(const half *input, half *output,
         int32_t channels, int32_t in_height, int32_t in_width, float bias, const KernelParams& kernelParams);

// default entry 0x1e
extern "C" void custom_entry(uint32_t* args, const KernelParams& kernelParams) {
    GRN((const half*)args[0], (half*)args[1],
        (int)args[2], (int)args[3], (int)args[4], ((float*)args)[5], kernelParams);
}

static float reciprocal_sqrt(float val)
{
#ifdef FAST_SQRT
    return (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val/16.f)))*0.25f;
#else
    return 1.f/sqrtf(val);
#endif
}

static float8 reciprocal_sqrt_float8(float8 val)
{
    float8 res = (float8)0;
#ifdef FAST_SQRT
    res[0] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[0]/16.f)))*0.25f;
    res[1] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[1]/16.f)))*0.25f;
    res[2] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[2]/16.f)))*0.25f;
    res[3] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[3]/16.f)))*0.25f;
    res[4] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[4]/16.f)))*0.25f;
    res[5] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[5]/16.f)))*0.25f;
    res[6] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[6]/16.f)))*0.25f;
    res[7] = (float)(__builtin_shave_sau_rqt_f16_l_r((half)(val[7]/16.f)))*0.25f;
#else
    res[0] = 1.f/sqrtf(val[0]);
    res[1] = 1.f/sqrtf(val[1]);
    res[2] = 1.f/sqrtf(val[2]);
    res[3] = 1.f/sqrtf(val[3]);
    res[4] = 1.f/sqrtf(val[4]);
    res[5] = 1.f/sqrtf(val[5]);
    res[6] = 1.f/sqrtf(val[6]);
    res[7] = 1.f/sqrtf(val[7]);
#endif

    return res;
}

static void grn_NCHW(const half* src_data, half* dst_data, int C, int W, float bias)
{
    const float eps = 1e-9f;
    bias += eps;

    int x = 0;
    for (; x <= W - 16; x += 16)
    {
        float8 variance0 = (float8)(bias);
        float8 variance1 = (float8)(bias);

        int c = 0;
        for (; c <= C - 4; c += 4)
        {
            const half8 psrc00 = *((const half8*)(src_data + (c + 0) * W + x));
            const half8 psrc01 = *((const half8*)(src_data + (c + 1) * W + x));
            const half8 psrc02 = *((const half8*)(src_data + (c + 2) * W + x));
            const half8 psrc03 = *((const half8*)(src_data + (c + 3) * W + x));
            const half8 psrc10 = *((const half8*)(src_data + (c + 0) * W + x + 8));
            const half8 psrc11 = *((const half8*)(src_data + (c + 1) * W + x + 8));
            const half8 psrc12 = *((const half8*)(src_data + (c + 2) * W + x + 8));
            const half8 psrc13 = *((const half8*)(src_data + (c + 3) * W + x + 8));

            const float8 fsrc00 = mvuConvert_float8(psrc00);
            const float8 fsrc01 = mvuConvert_float8(psrc01);
            const float8 fsrc02 = mvuConvert_float8(psrc02);
            const float8 fsrc03 = mvuConvert_float8(psrc03);
            const float8 fsrc10 = mvuConvert_float8(psrc10);
            const float8 fsrc11 = mvuConvert_float8(psrc11);
            const float8 fsrc12 = mvuConvert_float8(psrc12);
            const float8 fsrc13 = mvuConvert_float8(psrc13);

            variance0 += fsrc00 * fsrc00;
            variance0 += fsrc01 * fsrc01;
            variance0 += fsrc02 * fsrc02;
            variance0 += fsrc03 * fsrc03;

            variance1 += fsrc10 * fsrc10;
            variance1 += fsrc11 * fsrc11;
            variance1 += fsrc12 * fsrc12;
            variance1 += fsrc13 * fsrc13;
        }
        for (; c < C; c++)
        {
            const half8 psrc0 = *((const half8*)(src_data + c*W + x));
            const half8 psrc1 = *((const half8*)(src_data + c*W + x + 8));

            const float8 fsrc0 = mvuConvert_float8(psrc0);
            const float8 fsrc1 = mvuConvert_float8(psrc1);

            variance0 += fsrc0 * fsrc0;
            variance1 += fsrc1 * fsrc1;
        }

        variance0 = reciprocal_sqrt_float8(variance0);
        variance1 = reciprocal_sqrt_float8(variance1);

        c = 0;
        for (; c <= C - 4; c += 4)
        {
            const half8 psrc00 = *((const half8*)(src_data + (c + 0) * W + x));
            const half8 psrc01 = *((const half8*)(src_data + (c + 1) * W + x));
            const half8 psrc02 = *((const half8*)(src_data + (c + 2) * W + x));
            const half8 psrc03 = *((const half8*)(src_data + (c + 3) * W + x));
            const half8 psrc10 = *((const half8*)(src_data + (c + 0) * W + x + 8));
            const half8 psrc11 = *((const half8*)(src_data + (c + 1) * W + x + 8));
            const half8 psrc12 = *((const half8*)(src_data + (c + 2) * W + x + 8));
            const half8 psrc13 = *((const half8*)(src_data + (c + 3) * W + x + 8));

            half8* pdst0 = (half8*)(dst_data + (c + 0) * W + x);
            half8* pdst1 = (half8*)(dst_data + (c + 1) * W + x);
            half8* pdst2 = (half8*)(dst_data + (c + 2) * W + x);
            half8* pdst3 = (half8*)(dst_data + (c + 3) * W + x);

            pdst0[0] = psrc00 * mvuConvert_half8(variance0);
            pdst1[0] = psrc01 * mvuConvert_half8(variance0);
            pdst2[0] = psrc02 * mvuConvert_half8(variance0);
            pdst3[0] = psrc03 * mvuConvert_half8(variance0);

            pdst0[1] = psrc10 * mvuConvert_half8(variance1);
            pdst1[1] = psrc11 * mvuConvert_half8(variance1);
            pdst2[1] = psrc12 * mvuConvert_half8(variance1);
            pdst3[1] = psrc13 * mvuConvert_half8(variance1);
        }
        for (; c < C; c++)
        {
            const half8 psrc0 = *((const half8*)(src_data + c*W + x));
            const half8 psrc1 = *((const half8*)(src_data + c*W + x + 8));

            half8* pdst = (half8*)(dst_data + c*W + x);

            pdst[0] = psrc0 * mvuConvert_half8(variance0);
            pdst[1] = psrc1 * mvuConvert_half8(variance1);
        }
    }
    for (; x < W; x++)
    {
        const half* psrc = src_data;
        half* pdst = dst_data;

        float variance = bias;
        int c = 0;
        for (; c <= C - 4; c+=4)
        {
            variance += (float)(psrc[(c + 0) * W + x]) * (float)(psrc[(c + 0) * W + x]);
            variance += (float)(psrc[(c + 1) * W + x]) * (float)(psrc[(c + 1) * W + x]);
            variance += (float)(psrc[(c + 2) * W + x]) * (float)(psrc[(c + 2) * W + x]);
            variance += (float)(psrc[(c + 3) * W + x]) * (float)(psrc[(c + 3) * W + x]);
        }
        for (; c < C; c++)
        {
            variance += (float)(psrc[c*W + x]) * (float)(psrc[c*W + x]);
        }

        variance = reciprocal_sqrt(variance);

        c = 0;
        for (; c <= C - 4; c += 4)
        {
            pdst[(c + 0) * W + x] = (float)(psrc[(c + 0) * W + x]) * variance;
            pdst[(c + 1) * W + x] = (float)(psrc[(c + 1) * W + x]) * variance;
            pdst[(c + 2) * W + x] = (float)(psrc[(c + 2) * W + x]) * variance;
            pdst[(c + 3) * W + x] = (float)(psrc[(c + 3) * W + x]) * variance;
        }
        for (; c < C; c++)
        {
            pdst[c * W + x] = (float)(psrc[c * W + x]) * variance;
        }
    }
}

void GRN(const half *input, half *output,
          int32_t channels, int32_t in_height, int32_t in_width, float biasf, const KernelParams& kernelParams) {
    const ScheduleInfo& sinfo = kernelParams.scheduleInfo;
    const MemoryInfo& memoryInfo = kernelParams.memoryInfo;
    const DmaAlShaveWrapper& dmaAlShaveWrapper = kernelParams.dmaAlShaveWrapper;

    const int H = in_height;
    const int W = in_width;
    const int C = channels;

    const int startLine = sinfo.shaveId * H / sinfo.nShaves;
    const int endLine = (sinfo.shaveId + 1) * H / sinfo.nShaves;

    const half *psrc = input;
    half *pdst = output;

    const half bias = biasf;

    const int maxW = memoryInfo.availableCmxBytes/(2*C*sizeof(half));

    const int block_count = divRoundUp(W, maxW);
    const int width = std::min(W, divRoundUp(W, block_count));

    half* srcCmxBuf = reinterpret_cast<half *>(memoryInfo.cmxData);
    half* dstCmxBuf = reinterpret_cast<half *>(memoryInfo.cmxData + memoryInfo.availableCmxBytes/(2*sizeof(half)));

    half* src_buf = srcCmxBuf;
    half* dst_buf = dstCmxBuf;

    const int y0 = startLine;
    const int y1 = endLine;

    for (int x = 0; x < W; x += width )
    {
        const int x1 = std::min(x + width, W);
        const int length = x1 - x;

        const int buf_length = C * length * sizeof(half);
        const int buf_width = length * sizeof(half);
        const int data_stride = H*W*sizeof(half);

        for (int y = y0; y < y1; y++)
        {
            dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                          psrc + y*W + x, src_buf, buf_length, buf_width, buf_width, data_stride, buf_width);
            dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);

            grn_NCHW(src_buf, dst_buf, C, length, bias);

            dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                          dst_buf, pdst + y*W + x, buf_length, buf_width, buf_width, buf_width, data_stride);
            dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
        }
    }
}
