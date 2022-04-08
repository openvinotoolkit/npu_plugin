// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <float.h>
#include <moviVectorConvert.h>
#include <mv_types.h>

#include <param_max_pool.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void single_shave_max_pool(const struct MaxPoolParams* p) {
    half* in = (half*)(p->input.dataAddr);
    half* out = (half*)(p->output.dataAddr);

    int32_t* iDims = (int32_t*)p->input.dimsAddr;
    int32_t* oDims = (int32_t*)p->output.dimsAddr;

    // i32 version of i64 params
    int32_t kernel[2];
    int32_t strides[2];
    kernel[0] = (int32_t)p->kernelSize[0];
    kernel[1] = (int32_t)p->kernelSize[1];
    strides[0] = (int32_t)p->strides[0];
    strides[1] = (int32_t)p->strides[1];

    int N, C;    // in/out N,C are the same
    int iH, iW;  // input dims
    int oH, oW;  // output dims

    N = iDims[3];  // 'dims' are innermost-first, thus N is last dim

    uint32_t elStride;   // dist to next elem in same ch
    uint32_t iChStride;  // dist to same elem in next ch (input buf)
    uint32_t oChStride;  // dist to same elem in next ch (out buf)
    uint32_t iLnStride;  // in line stride [elements]
    uint32_t oLnStride;  // out line stride [elements]

    if (p->input.dimsOrder == 0x4321) {  // NCHW
        C = iDims[2];
        iH = iDims[1];
        oH = oDims[1];
        iW = iDims[0];
        oW = oDims[0];
        elStride = 1;
        iChStride = iW * iH;
        oChStride = oW * oH;
        iLnStride = iW;
        oLnStride = oW;
    } else {  // NHWC
        iH = iDims[2];
        oH = oDims[2];
        iW = iDims[1];
        oW = oDims[1];
        C = iDims[0];
        elStride = C;
        iChStride = 1;
        oChStride = 1;
        iLnStride = iW * C;
        oLnStride = oW * C;
    }

    uint32_t ix, iy;  // indices for reading input
    uint32_t ox, oy;  // indices for writing output
    uint32_t wx, wy;  // kernel window coords
    uint32_t kx, ky;  // kernel indices

    uint32_t kSize = kernel[0] * kernel[1];
    float norm = 1.0f / kSize;

    // for all lines
    for (iy = 0, oy = 0; iy < iH; iy += strides[0], oy++) {
        if (iy + kernel[0] > iH)
            break;  // no V-pad

        // current line
        ox = 0;

        for (ix = 0; ix < iLnStride; ix += strides[1] * elStride, ox++) {
            if (ix + kernel[1] * elStride > iLnStride)
                break;  // no H-pad

            // current pix for all channels
            for (uint32_t ch = 0; ch < C; ch++) {
                // Current output pixel filter
                float max = -FLT_MAX;
                for (ky = 0; ky < kernel[0]; ky++) {
                    for (kx = 0; kx < kernel[1]; kx++) {
                        max = MAX(max, in[(iy + ky) * iLnStride + ix + (kx * elStride) + ch * iChStride]);
                    }
                }

                out[oy * oLnStride + ch * oChStride + ox] = max;
            }
        }
    }
}

}  // extern "C"
}  // namespace shave_lib
}  // namespace nn
