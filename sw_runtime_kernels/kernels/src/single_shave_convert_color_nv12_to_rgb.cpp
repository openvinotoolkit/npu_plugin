//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <param_convert_color_nv12_to_rgb.h>

using namespace sw_params;

namespace nn {
namespace shave_lib {

static half clip(float a) {
    return static_cast<half>(std::min(std::max(a, 0.f), 255.0f));
}

static void calculate_output(half* R_cmx, half* G_cmx, half* B_cmx, const half* y_cmx, const half* uv_cmx,
                             size_t y_index, size_t uv_index) {
    float y_val = static_cast<float>(*(reinterpret_cast<const half*>(y_cmx) + y_index));
    float u_val = static_cast<float>(*(reinterpret_cast<const half*>(uv_cmx) + uv_index));
    float v_val = static_cast<float>(*(reinterpret_cast<const half*>(uv_cmx) + uv_index + 1));

    float c = y_val - 16.f;
    float d = u_val - 128.f;
    float e = v_val - 128.f;

    half r = clip(1.164f * c + 1.596f * e);               // R
    half g = clip(1.164f * c - 0.391f * d - 0.813f * e);  // G
    half b = clip(1.164f * c + 2.018f * d);               // B

    *(reinterpret_cast<half*>(R_cmx)) = r;
    *(reinterpret_cast<half*>(G_cmx)) = g;
    *(reinterpret_cast<half*>(B_cmx)) = b;
}

extern "C" {

void single_shave_convert_color_nv12_to_rgb(const struct ConvertColorNV12ToRGBParams* lParams) {
    enum { N_DIM = 3, H_DIM = 2, W_DIM = 1, C_DIM = 0 };

    half* y_values = (half*)(lParams->y_input.dataAddr);  // 0x1F000000
    half* uv_values = (half*)(lParams->uv_input.dataAddr);
    half* output = (half*)(lParams->output.dataAddr);  // 0x1F004000

    int64_t* p_y_stride = (int64_t*)(lParams->y_input.stridesAddr);
    int64_t* p_uv_stride = (int64_t*)(lParams->uv_input.stridesAddr);
    int64_t* p_out_stride = (int64_t*)(lParams->output.stridesAddr);

    int64_t color_format = lParams->rgbFormat;
    int32_t* yDims = (int32_t*)(lParams->y_input.dimsAddr);
    int32_t batch_size = yDims[N_DIM];
    int32_t image_h = yDims[H_DIM];
    int32_t image_w = yDims[W_DIM];
    int32_t stride_y = p_y_stride[N_DIM] / 8 / 2;
    int32_t stride_uv = p_uv_stride[N_DIM] / 8 / 2;

    int32_t stride_output = p_out_stride[N_DIM] / 8 / 2;
    int32_t bpp = 1;
    half* R_cmx;
    half* G_cmx;
    half* B_cmx;
    half* y_cmx;
    half* uv_cmx;

    for (int32_t batch = 0; batch < batch_size; batch++) {
        half* out = reinterpret_cast<half*>(output) + batch * stride_output;
        half* y_ptr = reinterpret_cast<half*>(y_values) + batch * stride_y;
        half* uv_ptr = reinterpret_cast<half*>(uv_values) + batch * stride_uv;

        if (color_format == 0) {
            R_cmx = out;
            G_cmx = R_cmx + bpp;
            B_cmx = G_cmx + bpp;
        } else {
            B_cmx = out;
            G_cmx = B_cmx + bpp;
            R_cmx = G_cmx + bpp;
        }

        for (size_t h = 0; h < image_h; h++) {
            y_cmx = y_ptr + h * image_w;
            uv_cmx = uv_ptr + (h / 2) * image_w;

            for (size_t w = 0; w < image_w; w++) {
                size_t y_index = w;
                size_t uv_index = (w / 2) * 2;

                calculate_output(R_cmx, G_cmx, B_cmx, y_cmx, uv_cmx, y_index, uv_index);

                R_cmx += 3 * bpp;
                G_cmx += 3 * bpp;
                B_cmx += 3 * bpp;
            }
        }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
