//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <param_pad.h>

#define bpp sizeof(half)

using namespace sw_params;

namespace nn {
namespace shave_lib {

template <typename VT, typename T>
static void copy_plane_impl(const T src[], T dst[], const int width, const int height, const int src_stride,
                            const int dst_stride) {
    static constexpr int veclen = sizeof(VT) / sizeof(T);
    uint32_t chunk = veclen * bpp;
    for (int w = 0; w < width;) {
        for (; w <= width - veclen; w += veclen) {
            for (int h = 0; h < height; h++) {
                memcpy_s(dst + w + h * dst_stride, chunk, src + w + h * dst_stride, chunk);
            }
        }

        if (w < width)
            w = width - veclen;
    }
}

__attribute__((noinline)) static void copy_plane(const half src[], half dst[], int width, int height, int src_stride,
                                                 int dst_stride) {
    if (width >= 8) {
        copy_plane_impl<half8>(src, dst, width, height, src_stride, dst_stride);
    } else if (width >= 4) {
        copy_plane_impl<half4>(src, dst, width, height, src_stride, dst_stride);
    } else {
        copy_plane_impl<half>(src, dst, width, height, src_stride, dst_stride);
    }
}

template <typename VT, typename T>
static void fill_edge_impl(T plane[], int height, int stride,
                           int padwidth,   // width of pad area
                           int padoffset,  // offset of pad area
                           int valoffset)  // offset of border value
{
    static const int veclen = sizeof(VT) / sizeof(T);
    T* src = plane + valoffset;
    uint32_t chunk = veclen * bpp;

    for (int w = 0; w < padwidth;) {
        for (; w <= padwidth - veclen; w += veclen) {
            T* dst = plane + padoffset + w;

            for (int h = 0; h < height; h++) {
                memcpy_s(dst + h * stride, chunk, src + h * stride, chunk);
            }
        }

        if (w < padwidth)
            w = padwidth - veclen;
    }
}

__attribute__((noinline)) static void fill_edge(half plane[], int height, int width, int pad_begin, int pad_end,
                                                int stride) {
    if (pad_begin >= 8) {
        fill_edge_impl<half8>(plane, height, stride, pad_begin, 0, pad_begin);
    } else if (pad_begin >= 4) {
        fill_edge_impl<half4>(plane, height, stride, pad_begin, 0, pad_begin);
    } else {
        fill_edge_impl<half>(plane, height, stride, pad_begin, 0, pad_begin);
    }

    int padoffset = width - pad_end;
    int valoffset = width - pad_end - 1;

    if (pad_end >= 8) {
        fill_edge_impl<half8>(plane, height, stride, pad_end, padoffset, valoffset);
    } else if (pad_end >= 4) {
        fill_edge_impl<half4>(plane, height, stride, pad_end, padoffset, valoffset);
    } else {
        fill_edge_impl<half>(plane, height, stride, pad_end, padoffset, valoffset);
    }
}

template <typename VT, typename T>
static void fill_plane_with_line_impl(const T line[], T plane[], const int height, const int width, const int stride) {
    static const int veclen = sizeof(VT) / sizeof(T);
    uint32_t chunk = veclen * bpp;
    for (int w = 0; w < width;) {
        // main: vectored by width
        for (; w <= width - veclen; w += veclen) {
            const T* src = line + w;
            T* dst = plane + w;

            for (int h = 0; h < height; h++) {
                memcpy_s(dst + h * stride, chunk, src, chunk);
            }
        }

        // tail: vectored as well
        if (w < width)
            w = width - veclen;
    }
}

__attribute__((noinline)) static void fill_plane_with_line(const half line[], half plane[], const int height,
                                                           const int width, const int stride) {
    if (width >= 8) {
        fill_plane_with_line_impl<half8>(line, plane, height, width, stride);
    } else if (width >= 4) {
        fill_plane_with_line_impl<half4>(line, plane, height, width, stride);
    } else {
        fill_plane_with_line_impl<half>(line, plane, height, width, stride);
    }
}

template <typename VT>
static VT vecinv(const VT& val);

template <>
half8 vecinv(const half8& val) {
    return {val[7], val[6], val[5], val[4], val[3], val[2], val[1], val[0]};
}

template <>
half4 vecinv(const half4& val) {
    return {val[3], val[2], val[1], val[0]};
}

template <>
half vecinv(const half& val) {
    return val;
}

template <typename VT, typename T>
static void fill_reflect_impl(T plane[], int height, int stride, int begin, int end, int inv0) {
    static const int veclen = sizeof(VT) / sizeof(T);
    uint32_t chunk = veclen * bpp;
    for (int w = begin; w < end;) {
        for (; w <= end - veclen; w += veclen) {
            T* src = plane + inv0 - (w - begin) - (veclen - 1);
            T* dst = plane + w;

            for (int h = 0; h < height; h++) {
                auto invvec = vecinv(*(VT*)&src[h * stride]);
                memcpy_s(dst + h * stride, chunk, &invvec, chunk);
            }
        }

        if (w < end)
            w = end - veclen;
    }
}

__attribute__((noinline)) static void fill_reflect(half plane[], int height, int width, int stride, int pad_begin,
                                                   int pad_end, int mode_offset) {
    if (pad_begin >= 8) {
        fill_reflect_impl<half8>(plane, height, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
    } else if (pad_begin >= 4) {
        fill_reflect_impl<half4>(plane, height, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
    } else {
        fill_reflect_impl<half>(plane, height, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
    }

    if (pad_end >= 8) {
        fill_reflect_impl<half8>(plane, height, stride, width - pad_end, width,
                                 width - pad_end - 1 - (1 - mode_offset));
    } else if (pad_end >= 4) {
        fill_reflect_impl<half4>(plane, height, stride, width - pad_end, width,
                                 width - pad_end - 1 - (1 - mode_offset));
    } else {
        fill_reflect_impl<half>(plane, height, stride, width - pad_end, width, width - pad_end - 1 - (1 - mode_offset));
    }
}

template <typename VT, typename T>
static void fill_reflect_plane_impl(T plane[], int width, int stride, int begin, int end, int inv1) {
    static const int veclen = sizeof(VT) / sizeof(T);
    uint32_t chunk = veclen * bpp;
    for (int w = 0; w < width;) {
        for (; w <= width - veclen; w += veclen) {
            for (int h = begin; h < end; h++) {
                memcpy_s(plane + w + h * stride, chunk, plane + w + (inv1 - (h - begin)) * stride, chunk);
            }
        }

        if (w < width)
            w = width - veclen;
    }
}

__attribute__((noinline)) static void fill_reflect_plane(half plane[], int height, int width, int stride, int pad_begin,
                                                         int pad_end, int mode_offset) {
    if (width >= 8) {
        fill_reflect_plane_impl<half8>(plane, width, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
        fill_reflect_plane_impl<half8>(plane, width, stride, height - pad_end, height,
                                       height - pad_end - 1 - (1 - mode_offset));
    } else if (width >= 4) {
        fill_reflect_plane_impl<half4>(plane, width, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
        fill_reflect_plane_impl<half4>(plane, width, stride, height - pad_end, height,
                                       height - pad_end - 1 - (1 - mode_offset));
    } else {
        fill_reflect_plane_impl<half>(plane, width, stride, 0, pad_begin, 2 * pad_begin - 1 + (1 - mode_offset));
        fill_reflect_plane_impl<half>(plane, width, stride, height - pad_end, height,
                                      height - pad_end - 1 - (1 - mode_offset));
    }
}

static void stridedMemcpy(uint8_t* src, uint8_t* dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                          uint32_t srcStride, uint32_t dstStride) {
    if (srcStride == 0)
        srcWidth = byteLength;
    if (dstStride == 0)
        dstWidth = byteLength;

    for (uint32_t si = 0, di = 0, length = byteLength; length > 0;) {
        const uint32_t chunk = std::min(std::min(srcWidth - si, dstWidth - di), length);
        memcpy_s(dst, chunk, src, chunk);

        si += chunk;
        di += chunk;
        src += chunk;
        dst += chunk;
        length -= chunk;

        if (si == srcWidth) {
            si = 0;
            src += srcStride;
        }

        if (di == dstWidth) {
            di = 0;
            dst += dstStride;
        }
    }
}

extern "C" {

void single_shave_pad(uint32_t lParams) {
    auto layerParams = reinterpret_cast<const PadParams*>(lParams);

    // Special DMA to copy layer params from physical DDR
    half* p_act_data = (half*)(layerParams->input.dataAddr);  // 0x1F000000
    half* p_act_out = (half*)(layerParams->output.dataAddr);  // 0x1F004000

    int64_t* p_in_stride = (int64_t*)(layerParams->input.stridesAddr);
    int64_t* p_out_stride = (int64_t*)(layerParams->output.stridesAddr);

    int32_t nElements = 1;
    int32_t* pDims = (int32_t*)(layerParams->input.dimsAddr);

    for (int i = 0; i < layerParams->input.numDims; i++) {
        nElements *= pDims[i];
    }

    float padValue = layerParams->pad_value;
    int pad0_begin = layerParams->pad_begin[0];
    int pad1_begin = layerParams->pad_begin[1];
    int pad2_begin = layerParams->pad_begin[2];
    int pad0_end = layerParams->pad_end[0];
    int pad1_end = layerParams->pad_end[1];
    int pad2_end = layerParams->pad_end[2];

    int inStride1 = p_in_stride[1] / 8;
    int inStride2 = p_in_stride[2] / 8;
    int outStride1 = p_out_stride[1] / 8;
    int outStride2 = p_out_stride[2] / 8;

    int inDim0 = pDims[0];
    int inDim1 = pDims[1];
    int inDim2 = pDims[2];

    int outDim0 = inDim0 + pad0_begin + pad0_end;
    int outDim1 = inDim1 + pad1_begin + pad1_end;
    int outDim2 = inDim2 + pad2_begin + pad2_end;

    int outDim0Aligned = ((outDim0 + 7) / 8) * 8;

    switch (layerParams->pad_mode) {
    case 0: {
        int maxPixels = outDim0 * outDim1 * outDim2;
        for (int i = 0; i < maxPixels; ++i) {
            ((half*)p_act_out)[i] = padValue;
        }

        int in_offset = 0;
        int out_offset = outStride2 * pad2_begin + (outDim0 * pad1_begin + pad0_begin) * bpp;

        for (int plane = 0; plane < inDim2; ++plane) {
            stridedMemcpy((uint8_t*)p_act_data + in_offset,  // src
                          (uint8_t*)p_act_out + out_offset,  // dst
                          inStride2,                         // bytelength
                          0,                                 // srcwidth
                          inStride1,                         // dstwidth
                          0,                                 // srcstride
                          (pad0_begin + pad0_end) * bpp);    // dststride

            in_offset += inStride2;
            out_offset += outStride2;
        }

        return;
    }
    case 1: {
        // init offset
        int in_offset = 0;
        int out_offset = outStride2 * pad2_begin + (outDim0 * pad1_begin + pad0_begin) * bpp;

        // process plane
        for (int plane = 0; plane < inDim2; ++plane) {
            // copy input
            stridedMemcpy((uint8_t*)p_act_data + in_offset,  // src
                          (uint8_t*)p_act_out + out_offset,  // dst
                          inStride2,                         // bytelength
                          0,                                 // srcwidth
                          inStride1,                         // dstwidth
                          0,                                 // srcstride
                          (pad0_begin + pad0_end) * bpp);    // dststride

            // fill edge
            fill_edge((half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp +
                              pad1_begin * outStride1 / bpp),  // plane
                      inDim1,                                  // height
                      outDim0,                                 // width
                      pad0_begin,                              // pad_begin
                      pad0_end,                                // pad_end
                      outDim0);                                // stride

            // padding by height: begin
            if (pad1_begin > 0) {
                fill_plane_with_line((half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp +
                                             pad1_begin * outStride1 / bpp),                        // line
                                     (half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp),  // plane
                                     pad1_begin,                                                    // height
                                     outDim0,                                                       // width
                                     outStride1 / bpp);                                             // stride
            }

            // padding by height: end
            if (pad1_end > 0) {
                fill_plane_with_line((half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp +
                                             (pad1_begin + inDim1 - 1) * outStride1 / bpp),  // line
                                     (half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp +
                                             (pad1_begin + inDim1) * outStride1 / bpp),  // plane
                                     pad1_end,                                           // height
                                     outDim0,                                            // width
                                     outStride1 / bpp);                                  // stride
            }

            in_offset += inStride2;
            out_offset += outStride2;
        }

        // padding first plane and last plane
        if (pad2_begin > 0) {
            for (int plane = 0; plane < pad2_begin; ++plane) {
                copy_plane((half*)(p_act_out + pad2_begin * outStride2 / bpp),  // src
                           (half*)(p_act_out + plane * outStride2 / bpp),       // dst
                           outDim0,                                             // width
                           outDim1,                                             // height
                           outStride1 / bpp,                                    // src_stride
                           outStride1 / bpp);                                   // dst_stride
            }
        }

        if (pad2_end > 0) {
            for (int plane = 0; plane < pad2_end; ++plane) {
                copy_plane((half*)(p_act_out + (outDim2 - pad2_end - 1) * outStride2 / bpp),      // src
                           (half*)(p_act_out + (outDim2 - pad2_end + plane) * outStride2 / bpp),  // dst
                           outDim0,                                                               // width
                           outDim1,                                                               // height
                           outStride1 / bpp,                                                      // src_stride
                           outStride1 / bpp);                                                     // dst_stride
            }
        }
        return;
    }
    case 2:
    case 3: {
        int mode_offset = (layerParams->pad_mode == 3) ? 1 : 0;

        if (pad0_begin + (1 - mode_offset) > inDim0 || pad0_end + (1 - mode_offset) > inDim0) {
            return;  // border is too wide
        }
        if (pad1_begin + (1 - mode_offset) > inDim1 || pad1_end + (1 - mode_offset) > inDim1) {
            return;  // border is too high
        }
        if (pad2_begin + (1 - mode_offset) > inDim2 || pad2_end + (1 - mode_offset) > inDim2) {
            return;  // border is too deep
        }

        // init offset
        int in_offset = 0;
        int out_offset = outStride2 * pad2_begin + (outDim0 * pad1_begin + pad0_begin) * bpp;

        // process plane
        for (int plane = 0; plane < inDim2; ++plane) {
            // copy input
            stridedMemcpy((uint8_t*)p_act_data + in_offset,  // src
                          (uint8_t*)p_act_out + out_offset,  // dst
                          inStride2,                         // bytelength
                          0,                                 // srcwidth
                          inStride1,                         // dstwidth
                          0,                                 // srcstride
                          (pad0_begin + pad0_end) * bpp);    // dststride

            in_offset += inStride2;
            out_offset += outStride2;

            // padding by width
            fill_reflect((half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp +
                                 pad1_begin * outStride1 / bpp),  // plane
                         inDim1,                                  // height
                         outDim0,                                 // width
                         outStride1 / bpp,                        // stride
                         pad0_begin,                              // pad_begin
                         pad0_end,                                // pad_end
                         mode_offset);                            // mode_offset

            // padding by height
            fill_reflect_plane((half*)(p_act_out + outStride2 * (plane + pad2_begin) / bpp),  // plane
                               outDim1, outDim0,                                              // height, width
                               outStride1 / bpp,                                              // stride
                               pad1_begin, pad1_end,                                          // pad_begin, pad_end
                               mode_offset);                                                  // mode_offset
        }

        // padding first plane and last plane
        if (pad2_begin > 0) {
            for (int plane = 0; plane < pad2_begin; ++plane) {
                copy_plane((half*)(p_act_out + (2 * pad2_begin - mode_offset - plane) * outStride2 / bpp),  // src
                           (half*)(p_act_out + plane * outStride2 / bpp),                                   // dst
                           outDim0,                                                                         // width
                           outDim1,                                                                         // height
                           outStride1 / bpp,   // src_stride
                           outStride1 / bpp);  // dst_stride
            }
        }

        if (pad2_end > 0) {
            for (int plane = 0; plane < pad2_end; ++plane) {
                copy_plane(
                        (half*)(p_act_out + (outDim2 - pad2_end - plane + mode_offset - 2) * outStride2 / bpp),  // src
                        (half*)(p_act_out + (outDim2 - pad2_end + plane) * outStride2 / bpp),                    // dst
                        outDim0,            // width
                        outDim1,            // height
                        outStride1 / bpp,   // src_stride
                        outStride1 / bpp);  // dst_stride
            }
        }

        return;
    }
    }

    return;
}
}
}  // namespace shave_lib
}  // namespace nn
