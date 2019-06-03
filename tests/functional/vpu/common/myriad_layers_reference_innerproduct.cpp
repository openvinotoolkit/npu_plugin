// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

const std::string inner_product_param = "out-size";

static void kchw_to_hwck(const uint16_t* src,
                         uint16_t* dst,
                         size_t dimx,
                         size_t dimy,
                         size_t dimz) {
    for (size_t x = 0 ; x < dimx; ++x) {
        for (size_t y = 0 ; y < dimy; ++y) {
            for (size_t z = 0 ; z < dimz; ++z) {
                size_t input = x + dimx * (y + dimy * z);
                size_t output = z + dimz * (y + dimy * x);
                dst[output] = src[input];
            }
        }
    }
}

void ref_innerproduct_wrap(const InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const uint16_t *weights,
                      size_t weightsSize,
                      size_t biasSize,
                      const ParamsStruct& params)
{
    uint32_t OC = 1;
    if (!params.empty()) {
        auto iter = params.find(inner_product_param);
        if (iter != params.end()) {
            OC = std::stol(iter->second);
        }
    }
    ref_innerproduct(src, dst, weights, weightsSize, biasSize, OC);
}

void ref_innerproduct(const Blob::Ptr src,
                      Blob::Ptr dst,
                      const uint16_t *weights,
                      size_t weightsSize,
                      size_t biasSize,
                      uint32_t OC) {

    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_GT(weightsSize, 0);
    size_t IW = 1;
    size_t IH = 1;
    size_t IC = 1;
    size_t I_N = 1;
    auto tensorDesc = src->getTensorDesc();
    auto dims = tensorDesc.getDims();
    switch(tensorDesc.getLayout())
    {
        case NCHW:
        case NHWC:
            IW = dims[3];
            IH = dims[2];
            IC = dims[1];
            I_N = dims[0];
            break;
        case NC:
            I_N = dims[1];
            IC  = dims[0];
            break;
        case HW:
            IH = dims[0];
            IW  = dims[1];
            break;
    }
    const uint16_t *src_data = static_cast<uint16_t*>(src->buffer());
    const uint16_t *weights_data = weights;
    uint16_t *dst_data = dst->buffer();

    uint16_t *weights_hwck = new uint16_t [IW * IH * IC * OC];
    const uint16_t *bias_data = nullptr;
    if (biasSize) {
        ASSERT_EQ(IW*IH*IC*OC + OC , weightsSize + biasSize);
        bias_data = &weights[IW*IH*IC*OC];
    }
    if (tensorDesc.getLayout() == NCHW ||
        tensorDesc.getLayout() == NHWC) {
        ASSERT_NE(weights_hwck, nullptr);
        kchw_to_hwck(weights_data, weights_hwck, (IW * IH), IC, OC);
        for (size_t on = 0; on < I_N; on++) {
            size_t offset = OC * on;
            for (size_t oc = 0; oc < OC; oc++) {
                float sum_f = 0.0f;
                if (bias_data)
                    sum_f = PrecisionUtils::f16tof32(bias_data[oc]);

                for (size_t ic = 0; ic < IC; ic++) {
                    for (size_t kh = 0; kh < IH; kh++) {
                        for (size_t  kw = 0; kw < IW; kw++) {
                            size_t iidx = ic * IH * IW + kh * IW + kw + on * IH * IW * IC;
                            size_t widx = ic * IH * IW + kh * IW + kw;
                            float mult = (PrecisionUtils::f16tof32(src_data[iidx]) * PrecisionUtils::f16tof32(weights_hwck[widx * OC + oc]));
                            sum_f = sum_f + mult;
                        }
                    }
                }
                dst_data[oc + offset] = PrecisionUtils::f32tof16(sum_f);
            }
        }
    } else if (tensorDesc.getLayout() == HW) {
        for (size_t kh = 0; kh < IH; kh++) {
            for (size_t oc = 0; oc < OC; oc++) {
                float sum_f = 0.0f;
                if (bias_data)
                    sum_f = PrecisionUtils::f16tof32(bias_data[oc]);
                for (size_t  kw = 0; kw < IW; kw++) {
                    size_t iidx = kh * IW + kw;
                    float mult = (PrecisionUtils::f16tof32(src_data[iidx]) * PrecisionUtils::f16tof32(weights_data[oc * IW + kw]));
                    sum_f = sum_f + mult;
                }
                dst_data[oc + kh * OC] = PrecisionUtils::f32tof16(sum_f);
            }
        }
    }
    delete[] weights_hwck;
}
