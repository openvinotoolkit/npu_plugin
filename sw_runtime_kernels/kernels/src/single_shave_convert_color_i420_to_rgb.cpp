//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <param_convert_color_i420_to_rgb.h>

namespace nn {
namespace shave_lib {

static half clip(float a) {
    return static_cast<half>(std::min(std::max(a, 0.f), 255.0f));
}

static void calculate_output(half* R_cmx, half* G_cmx, half* B_cmx, const half* y_cmx, const half* u_cmx,
                             const half* v_cmx, size_t y_index, size_t uv_index) {
    float y_val = static_cast<float>(*(reinterpret_cast<const half*>(y_cmx) + y_index));
    float u_val = static_cast<float>(*(reinterpret_cast<const half*>(u_cmx) + uv_index));
    float v_val = static_cast<float>(*(reinterpret_cast<const half*>(v_cmx) + uv_index));

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
void single_shave_convert_color_i420_to_rgb(const struct ConvertColorI420ToRGBParams* params) {
    enum { N_DIM = 3, H_DIM = 2, W_DIM = 1, C_DIM = 0 };

    half* y_values = (half*)params->y_input.dataAddr;
    half* u_values = (half*)params->u_input.dataAddr;
    half* v_values = (half*)params->v_input.dataAddr;
    half* output = (half*)params->output.dataAddr;

    int64_t color_format = params->outFmt;

    int32_t* y_dims = (int32_t*)(params->y_input.dimsAddr);

    size_t batch_size = y_dims[N_DIM];
    size_t image_h = y_dims[H_DIM];
    size_t image_w = y_dims[W_DIM];
    size_t stride_y = image_w * image_h;
    size_t stride_uv = image_w / 2 * image_h / 2;
    size_t stride_output = image_h * image_w * 3;

    const auto y_src = y_values;
    auto u_src = u_values;
    auto v_src = v_values;

    int32_t bpp = 1;
    half* R_cmx;
    half* G_cmx;
    half* B_cmx;
    half* y_cmx;
    half* u_cmx;
    half* v_cmx;

    for (size_t batch = 0; batch < batch_size; batch++) {
        half* out = reinterpret_cast<half*>(output) + batch * stride_output;
        half* y_ptr = reinterpret_cast<half*>(y_src) + batch * stride_y;
        half* u_ptr = reinterpret_cast<half*>(u_src) + batch * stride_uv;
        half* v_ptr = reinterpret_cast<half*>(v_src) + batch * stride_uv;

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
            u_cmx = u_ptr + (h / 2) * (image_w / 2);
            v_cmx = v_ptr + (h / 2) * (image_w / 2);

            for (size_t w = 0; w < image_w; w++) {
                size_t y_index = w;
                size_t uv_index = w / 2;

                calculate_output(R_cmx, G_cmx, B_cmx, y_cmx, u_cmx, v_cmx, y_index, uv_index);

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
