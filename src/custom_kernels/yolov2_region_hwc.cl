// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant static half log_2_e = (half)1.442695040888963; // log2(exp(1.0))

#define ALLOW_EARLY_RETURN 1

__kernel void __dma_preload_region_hwc(
    __global const half* restrict src,
    __global       half* restrict _0,
    __local        half* restrict local_src,
    __local        half* restrict _1,
    int W,       /* 13 */
    int H,       /* 13 */
    int classes, /* 20 */
    int coords,  /* 4  */
    int num      /* 5  */
    )
{
    const int local_C = classes + coords + 1;
    const int c = get_group_id(1)*local_C;
    const int h = get_group_id(0);
    const int C = local_C*num;

    WorkGroupDmaCreateStrideTransaction(
        src + h*W*C + c, // src
        local_src, // dst
        local_C*sizeof(half), // src_width,
        local_C*sizeof(half), // dst_width,
        C*sizeof(half), // src_stride,
        local_C*sizeof(half), // dst_stride,
        local_C*W*sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_region_hwc(
    __global       half* restrict _0,
    __global       half* restrict dst,
    __local        half* restrict _1,
    __local  const half* restrict local_dst,
    int W,       /* 13 */
    int H,       /* 13 */
    int classes, /* 20 */
    int coords,  /* 4  */
    int num      /* 5  */
    )
{
    // Region always outputs in CHW layout; same as postwrite_chw
    const int local_C = classes + coords + 1;
    const int c = get_group_id(1)*local_C;
    const int h = get_group_id(0);

    WorkGroupDmaCreateStrideTransaction(
        local_dst, // src
        dst + c*H*W + h*W, // dst
        W*sizeof(half), // src_width,
        W*sizeof(half), // dst_width,
        W*sizeof(half), // src_stride,
        W*H*sizeof(half), // dst_stride,
        W*local_C*sizeof(half), // size
        0);
}

static void inline logistic_activate_hwc(__local const half* restrict src,
                                         __local       half* restrict dst,
                                         int offset,
                                         int stride)
{
    half val = src[offset];
    val = 1.0h / (1.0h + exp2(val * -log_2_e));
    dst[offset*stride] = val;
}

__kernel void region_hwc(
    __global       half* restrict src_data,
    __global       half* restrict dst_data,
    __local  const half* restrict local_src,
    __local        half* restrict local_dst,
    int W,       /* 13 */
    int H,       /* 13 */
    int classes, /* 20 */
    int coords,  /* 4  */
    int num      /* 5  */
    )
{
    const int w = get_local_id(0);

#if ALLOW_EARLY_RETURN
    if (w >= W) return;
#endif

    const int local_C = classes + coords + 1;

    __local const half *restrict src = local_src + w*local_C;
    __local       half *restrict dst = local_dst + w;

    const int stride = W;
    logistic_activate_hwc(src, dst, 0, stride);
    logistic_activate_hwc(src, dst, 1, stride);

    //copy plane 2 and 3
    dst[2*stride] = src[2];
    dst[3*stride] = src[3];

    logistic_activate_hwc(src, dst, 4, stride);

    src += coords + 1;
    dst += (coords + 1)*stride;

    half max_val = src[0];
    #pragma unroll 4
    for (int c = 0; c < classes; c++) {
        max_val = max(max_val, src[c]);
    }

    half expSum = 0.0h;
    #pragma unroll 4
    for (int c = 0; c < classes; c++)
    {
        const half e = src[c] - max_val;
        const half tmp = exp2(e * log_2_e);
        dst[c*stride] = tmp;
        expSum += tmp;
    }

    const half invExpSum = 1.0h / expSum;
    #pragma unroll 4
    for (int c = 0; c < classes; c++)
    {
        dst[c*stride] *= invExpSum;
    }
}
