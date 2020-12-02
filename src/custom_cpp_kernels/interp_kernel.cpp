// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_params.h>

#include <moviVectorTypes.h>
#include <moviVectorUtils.h>
#include <moviVectorFunctions.h>

#include <math.h>
#include <algorithm>

static void interpHWC(const half*, half*, int, int, int,
                      int, int, int, int, bool, const KernelParams&);

static void interpCHW(const half*, half*, int, int, int,
                      int, int, int, int, bool, const KernelParams&);

// default entry 0x1e
extern "C" void custom_entry(uint32_t* args, const KernelParams& kernelParams) {
    if(args[10]){
        interpHWC((const half*)args[0], (half*)args[1], (int)args[2], (int)args[3], (int)args[4],
                  (int)args[5], (int)args[6], (int)args[7], (int)args[8], (bool)args[9], kernelParams);
    } else {
        interpCHW((const half*)args[0], (half*)args[1], (int)args[2], (int)args[3], (int)args[4],
                  (int)args[5], (int)args[6], (int)args[7], (int)args[8], (bool)args[9], kernelParams);
    }
}

//--------------------- HWC ---------------------

void interpolationHWC(half* psrc0, half* psrc1, half* pdst, int OW, int IW, int C,
                      float rw, float h_lambda0, float h_lambda1)
{
    for (int w = 0; w < OW; w++)
    {
        float fw = rw * w;
        int iw0 = static_cast<int>(fw);
        int iw1 = (iw0 < IW - 1) ? iw0 + 1 : iw0;

        float w_lambda0 = fw - iw0;
        float w_lambda1 = 1.0f - w_lambda0;

        half8 hl0 = mvuConvert_half(h_lambda0);
        half8 hl1 = mvuConvert_half(h_lambda1);
        half8 wl0 = mvuConvert_half(w_lambda0);
        half8 wl1 = mvuConvert_half(w_lambda1);

        for (int c = 0; c < C; c += 32)
        {
            half8 src0_00 = *((half8*)(psrc0 + iw0*C + c));
            half8 src0_01 = *((half8*)(psrc0 + iw1*C + c));
            half8 src0_10 = *((half8*)(psrc1 + iw0*C + c));
            half8 src0_11 = *((half8*)(psrc1 + iw1*C + c));

            half8 src1_00 = *((half8*)(psrc0 + iw0*C + c + 8));
            half8 src1_01 = *((half8*)(psrc0 + iw1*C + c + 8));
            half8 src1_10 = *((half8*)(psrc1 + iw0*C + c + 8));
            half8 src1_11 = *((half8*)(psrc1 + iw1*C + c + 8));

            half8 src2_00 = *((half8*)(psrc0 + iw0*C + c + 16));
            half8 src2_01 = *((half8*)(psrc0 + iw1*C + c + 16));
            half8 src2_10 = *((half8*)(psrc1 + iw0*C + c + 16));
            half8 src2_11 = *((half8*)(psrc1 + iw1*C + c + 16));

            half8 src3_00 = *((half8*)(psrc0 + iw0*C + c + 24));
            half8 src3_01 = *((half8*)(psrc0 + iw1*C + c + 24));
            half8 src3_10 = *((half8*)(psrc1 + iw0*C + c + 24));
            half8 src3_11 = *((half8*)(psrc1 + iw1*C + c + 24));

            half8 result0 = hl1 * (wl1 * src0_00 + wl0 * src0_01) +
                            hl0 * (wl1 * src0_10 + wl0 * src0_11);

            half8 result1 = hl1 * (wl1 * src1_00 + wl0 * src1_01) +
                            hl0 * (wl1 * src1_10 + wl0 * src1_11);

            half8 result2 = hl1 * (wl1 * src2_00 + wl0 * src2_01) +
                            hl0 * (wl1 * src2_10 + wl0 * src2_11);

            half8 result3 = hl1 * (wl1 * src3_00 + wl0 * src3_01) +
                            hl0 * (wl1 * src3_10 + wl0 * src3_11);

            *((half8*)(pdst + w * C + c + 0))  = result0;
            *((half8*)(pdst + w * C + c + 8))  = result1;
            *((half8*)(pdst + w * C + c + 16)) = result2;
            *((half8*)(pdst + w * C + c + 24)) = result3;
        }
    }
}

#define USE_DMA 1

void interpHWC(const half* input, half* output,
               int in_channels, int input_height, int input_width,
               int output_height, int output_width,
               int input_stride, int output_stride,
               bool align_corners, const KernelParams& kernelParams)
{
    const ScheduleInfo& sinfo = kernelParams.scheduleInfo;
    const MemoryInfo& memoryInfo = kernelParams.memoryInfo;
    const DmaAlShaveWrapper& dmaAlShaveWrapper = kernelParams.dmaAlShaveWrapper;

    int OH = output_height;
    int OW = output_width;

    int IH = input_height;
    int IW = input_width;

    int C = in_channels;
    int in_stride  = input_stride;
    int out_stride = output_stride;

    const half* psrc = input;
    half* pdst = output;

    int max_C = (memoryInfo.availableCmxBytes - 2*31*sizeof(half))/((2*IW + OW)*sizeof(half));
    if (max_C <= 0)
    {
        return;
    }

    int block_count = (C + max_C - 1)/max_C;
    int channels = (C + block_count - 1)/block_count;

    channels = std::min(C, channels);

    half* src_buf0 = (half*) memoryInfo.cmxData;
    half* src_buf1 = ((half*) memoryInfo.cmxData) + IW*channels + 31;
    half* dst_buf  = ((half*) memoryInfo.cmxData) + 2*(IW*channels + 31);

    int y0 = sinfo.shaveId * OH / sinfo.nShaves;
    int y1 = (sinfo.shaveId + 1) * OH / sinfo.nShaves;

    if (y0 >= OH || y0 == y1)
        return;

    const float rh = (OH > 1 && align_corners) ? static_cast<float>(IH - 1) / (OH - 1) : static_cast<float>(IH) / OH;
    const float rw = (OW > 1 && align_corners) ? static_cast<float>(IW - 1) / (OW - 1) : static_cast<float>(IW) / OW;

    for (int c = 0; c < C; c += channels )
    {
        int c1 = std::min(c + channels, C);
        int length = c1 - c;

        uint32_t in_buf_length = IW*length*sizeof(half);
        uint32_t in_data_stride = in_stride*sizeof(half);

        uint32_t out_buf_length = OW*length*sizeof(half);
        uint32_t out_data_stride = out_stride*sizeof(half);
        uint32_t buf_width = length*sizeof(half);

        int p_ih0 = -1;
        int p_ih1 = -1;

        for (int y = y0; y < y1; y++)
        {
            float fh = rh*y;
            int ih0 = static_cast<int>(fh);
            int ih1 = (ih0 < IH - 1) ? ih0 + 1 : ih0;

            float h_lambda0 = fh - ih0;
            float h_lambda1 = 1.0f - h_lambda0;

            if (p_ih0 != ih0 || p_ih1 != ih1)
            {
#if USE_DMA
                // TODO this was previously 2 chained DMAs. Can't use a single strided DMA because at the end the last line is duplicated
                dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,(uint8_t*)(psrc + ih0*IW*in_stride + c), (uint8_t*)(src_buf0), in_buf_length,
                          buf_width, buf_width, in_data_stride, buf_width);
                dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
                dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd, (uint8_t*)(psrc + ih1*IW*in_stride + c), (uint8_t*)(src_buf1), in_buf_length,
                          buf_width, buf_width, in_data_stride, buf_width);
                dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
#else
                memcpy((uint8_t*)(src_buf0), (uint8_t*)(psrc + ih0*IW*in_stride + c), in_buf_length);
                memcpy((uint8_t*)(src_buf1), (uint8_t*)(psrc + ih1*IW*in_stride + c), in_buf_length);
#endif
            }

            p_ih0 = ih0;
            p_ih1 = ih1;

            interpolationHWC(src_buf0, src_buf1, dst_buf, OW, IW, length, rw, h_lambda0, h_lambda1);

#if USE_DMA
            dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                          (uint8_t*)(dst_buf), (uint8_t*)(pdst + y*OW*out_stride + c), out_buf_length, buf_width, buf_width,
                                          buf_width, out_data_stride);
            dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
#else
            memcpy((uint8_t*)(pdst + y*OW*out_stride + c), (uint8_t*)(dst_buf), out_buf_length);
#endif
        }
    }
}

//--------------------- CHW ---------------------

void interpolationCHW(half* psrc0, half* psrc1, half* pdst, half* w_lambda, ushort* ind, half* row_buf, int OW, int IW, int C,
                      float h_lambda0, float h_lambda1, int ow_last_interpolate)
{
    half8 hl0 = mvuConvert_half(h_lambda0);
    half8 hl1 = mvuConvert_half(h_lambda1);

    for(int c = 0; c < C; c++)
    {
        for(int w = 0; w < IW; w += 8)
        {
            half8 vsrc0 = *((half8*)(psrc0 + c*IW + w));
            half8 vsrc1 = *((half8*)(psrc1 + c*IW + w));

            half8 result = hl1*vsrc0 + hl0*vsrc1;

            *((half8*)(row_buf + w)) = result;
        }

        for(int w = 0; w <= ow_last_interpolate; w += 8)
        {
            half8 s0 = {row_buf[ind[w + 0]], row_buf[ind[w + 1]], row_buf[ind[w + 2]], row_buf[ind[w + 3]],
                        row_buf[ind[w + 4]], row_buf[ind[w + 5]], row_buf[ind[w + 6]], row_buf[ind[w + 7]]};
            half8 s1 = {row_buf[ind[w + 0] + 1], row_buf[ind[w + 1] + 1], row_buf[ind[w + 2] + 1], row_buf[ind[w + 3] + 1],
                        row_buf[ind[w + 4] + 1], row_buf[ind[w + 5] + 1], row_buf[ind[w + 6] + 1], row_buf[ind[w + 7] + 1]};

            half8 wl0 = *((half8*)(w_lambda + w));
            half8 wl1 = 1.f - wl0;

            half8 result = wl1 * s0 + wl0 * s1;

            *((half8*)(pdst + c*OW + w)) = result;
        }
        //calc the last
        for (int w = ow_last_interpolate + 1; w < OW; w++)
        {
            pdst[(c + 0)*OW + w] = row_buf[IW - 1];
        }
    }
}

void interpolationCHW_2x(half* psrc0, half* psrc1, half* pdst, half* w_lambda, half* row_buf, int OW, int IW, int C,
                         float h_lambda0, float h_lambda1, bool align_corners, int ow_last_interpolate)
{
    half8 hl0 = mvuConvert_half(h_lambda0);
    half8 hl1 = mvuConvert_half(h_lambda1);

    int ow_first_interpolate = align_corners? 1 : 0;
    int ow_offset = align_corners? -1 : 0;

    for(int c = 0; c < C; c++)
    {
        for(int w = 0; w < IW; w += 8)
        {
            half8 vsrc0 = *((half8*)(psrc0 + c*IW + w));
            half8 vsrc1 = *((half8*)(psrc1 + c*IW + w));

            half8 result = hl1*vsrc0 + hl0*vsrc1;

            *((half8*)(row_buf + w)) = result;
        }

        pdst[c*OW] = row_buf[0];

        for(int w = ow_first_interpolate; w <= ow_last_interpolate; w += 8)
        {
            ushort4 v0 = *(ushort4*)(&row_buf[(w + ow_offset) / 2 + 0]);
            ulong4 v0_0 = mvuConvert_ulong4(v0);
            ushort8 v0_1 = mvuAs_ushort8(v0_0) + mvuAs_ushort8(v0_0 << 16);

            ushort4 v1 = *(ushort4*)(&row_buf[(w + ow_offset) / 2 + 1]);
            ulong4 v1_0 = mvuConvert_ulong4(v1);
            ushort8 v1_1 = mvuAs_ushort8(v1_0) + mvuAs_ushort8(v1_0 << 16);

            half8 wl0 = *((half8*)(w_lambda + w));
            half8 wl1 = 1.f - wl0;

            half8 result = wl1 * (mvuAs_half8(v0_1)) + wl0 * mvuAs_half8(v1_1);

            *((half8*)(pdst + c * OW + w)) = result;

        }
        //calc the last
        for (int w = ow_last_interpolate + 1; w < OW; w++)
        {
            pdst[(c + 0)*OW + w] = row_buf[IW - 1];
        }
    }
}

void interpCHW(const half* input, half* output,
               int in_channels, int input_height, int input_width,
               int output_height, int output_width,
               int input_stride, int output_stride,
               bool align_corners, const KernelParams& kernelParams)
{
    const ScheduleInfo& sinfo = kernelParams.scheduleInfo;
    const MemoryInfo& memoryInfo = kernelParams.memoryInfo;
    const DmaAlShaveWrapper& dmaAlShaveWrapper = kernelParams.dmaAlShaveWrapper;

    int OH = output_height;
    int OW = output_width;

    int IH = input_height;
    int IW = input_width;

    int C = in_channels;
    int in_stride  = input_stride;
    int out_stride = output_stride;

    const half* psrc = input;
    half* pdst = output;

    int max_C = (memoryInfo.availableCmxBytes - (IW + OW + 7*5)*sizeof(half) - (OW + 7)*sizeof(ushort))/((2*IW + OW)*sizeof(half));
    if (max_C <= 0)
    {
        return;
    }

    int block_count = (C + max_C - 1)/max_C;
    int channels = (C + block_count - 1)/block_count;

    channels = std::min(C, channels);

    half* src_buf0 = (half*) memoryInfo.cmxData;                          //[IW*channels + 7]
    half* src_buf1 = ((half*) memoryInfo.cmxData) + IW*channels + 7;      //[IW*channels + 7]
    half* dst_buf  = ((half*) memoryInfo.cmxData) + 2*(IW*channels + 7);  //[OW*channels + 7]

    half* lambda = dst_buf + OW*channels + 7; //[OW + 7]
    ushort* ind = (ushort*)(lambda + OW + 7); //[OW + 7]

    half* row_buf = (half*)(ind + OW + 7); //[IW + 7]

    int y0 = sinfo.shaveId * OH / sinfo.nShaves;
    int y1 = (sinfo.shaveId + 1) * OH / sinfo.nShaves;

    if (y0 >= OH || y0 == y1)
        return;

    const float rh_0 = (OH > 1 && align_corners) ? (IH-1): IH;
    const float rh_1 = (OH > 1 && align_corners) ? (OH-1): OH;

    const float rw_0 = (OW > 1 && align_corners) ? (IW-1): IW;
    const float rw_1 = (OW > 1 && align_corners) ? (OW-1): OW;

    const float rh = static_cast<float>(rh_0) / rh_1;
    const float rw = static_cast<float>(rw_0) / rw_1;

    /* ow_last_interpolate: max{oh: oh is interger, floor(oh * rw) <= IW - 2} */
    const float rw_inv = static_cast<float>(rw_1) / rw_0;
    int ow_last_interpolate = ceil((IW - 1) * rw_inv) - 1;

    for (int w = 0; w < OW; w++)
    {
        float fw = (float)(rw * w);
        ushort iw = static_cast<ushort>(fw);
        float wl = fw - iw;

        ind[w] = iw;
        lambda[w] = mvuConvert_half(wl);
    }

    for (int c = 0; c < C; c += channels )
    {
        int c1 = std::min(c + channels, C);
        int length = c1 - c;

        uint32_t in_buf_length  = IW*length*sizeof(half);
        uint32_t in_data_stride = in_stride*IH*sizeof(half);
        uint32_t in_buf_width   = IW*sizeof(half);

        int p_ih0 = -1;
        int p_ih1 = -1;

        for (int y = y0; y < y1; y++)
        {
            float fh = rh*y;
            int ih0 = static_cast<int>(fh);
            int ih1 = (ih0 < IH - 1) ? ih0 + 1 : ih0;

            float h_lambda0 = fh - ih0;
            float h_lambda1 = 1.0f - h_lambda0;

            if (p_ih0 != ih0 || p_ih1 != ih1)
            {
                dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                              (uint8_t*)(psrc + c*IH*in_stride + ih0*in_stride), (uint8_t*)(src_buf0),
                                              in_buf_length, in_buf_width, in_buf_width, in_data_stride, in_buf_width);
                dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
                dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                              (uint8_t*)(psrc + c*IH*in_stride + ih1*in_stride), (uint8_t*)(src_buf1),
                                              in_buf_length, in_buf_width, in_buf_width, in_data_stride, in_buf_width);
                dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
            }

            if (IW * 2 == OW && IH * 2 == OH) {
                interpolationCHW_2x(src_buf0, src_buf1, dst_buf, lambda, row_buf, OW, IW, length, h_lambda0, h_lambda1, align_corners, ow_last_interpolate);
            } else {
                interpolationCHW(src_buf0, src_buf1, dst_buf, lambda, ind, row_buf, OW, IW, length, h_lambda0, h_lambda1, ow_last_interpolate);
            }
            dmaAlShaveWrapper.startStride(dmaAlShaveWrapper.dmaAlShaveHnd,
                                          (uint8_t*)(dst_buf), (uint8_t*)(pdst + c*OH*out_stride + y*out_stride),
                                          OW*length*sizeof(half),
                                          OW*sizeof(half), OW*sizeof(half),
                                          OW*sizeof(half), out_stride*OH*sizeof(half));

            dmaAlShaveWrapper.wait(dmaAlShaveWrapper.dmaAlShaveHnd);
        }
    }
}
