// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (5.e-3f)

using namespace InferenceEngine;

void ref_scale(const InferenceEngine::Blob::Ptr src,
                      const uint16_t *weights,
                      InferenceEngine::Blob::Ptr dst,
                      bool bias)
{
    ASSERT_NE(src, nullptr);
    ASSERT_NE(weights, nullptr);
    ASSERT_NE(dst, nullptr);
    size_t IW = src->dims()[0];
    size_t IH = src->dims()[1];
    size_t IC = src->dims()[2];

    const uint16_t *src_data = src->buffer();
    const uint16_t *bias_data = weights + IC;
    uint16_t *dst_data = dst->buffer();
    for (size_t ic = 0; ic < IC; ic++) {
        float val = 0.0f;
        if (bias)
            val = PrecisionUtils::f16tof32(bias_data[ic]);
        for (size_t kh = 0; kh < IH; kh++) {
            for (size_t  kw = 0; kw < IW; kw++) {
                size_t iidx = ic + kw * IC + kh * IC * IW;
                float res = val + PrecisionUtils::f16tof32(src_data[iidx]) *
                        PrecisionUtils::f16tof32(weights[ic]);
                dst_data[iidx] = PrecisionUtils::f32tof16(res);
            }
        }
    }
}

typedef std::tuple<Dims, bool> TestScaleShift;

class myriadLayersTestsScale_nightly: public myriadLayersTests_nightly,
                              public testing::WithParamInterface<TestScaleShift> {
};

TEST_P(myriadLayersTestsScale_nightly, TestsScale)
{
    tensor_test_params p = std::get<0>(::testing::WithParamInterface<TestScaleShift>::GetParam());
    bool biasAdd = std::get<1>(::testing::WithParamInterface<TestScaleShift>::GetParam());
    size_t sz_weights = p.c;
    size_t sz_bias = p.c * biasAdd;
    size_t sz = sz_weights + sz_bias;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(sz));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();
    IN_OUT_desc inpt = {{p.n, p.c, p.h, p.w}};
    SetInputTensors(inpt);
    SetOutputTensors(inpt);
    NetworkInit("ScaleShift",
                nullptr,
                sz_weights * sizeof(uint16_t),
                sz_bias * sizeof(uint16_t),
                weights_ptr,
                InferenceEngine::Precision::FP16);
    ASSERT_TRUE(Infer());
    ref_scale(_inputMap.begin()->second, weights, _refBlob, biasAdd);
    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_inputScaleTensors = {
    {{1, 1, 16, 8}},
    {{1, 4, 8, 16}},
    {{1, 44, 88, 16}},
    {{1, 16, 32, 32}},
    {{1, 512, 7, 7}}
};

static std::vector<bool> s_inputBiasScale = {
    false,
    true
};

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsScale_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inputScaleTensors),
            ::testing::ValuesIn(s_inputBiasScale)));
