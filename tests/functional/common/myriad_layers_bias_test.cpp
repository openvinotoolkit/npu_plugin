// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (1.e-3f)

using namespace InferenceEngine;

void ref_bias(const InferenceEngine::Blob::Ptr src1,
              const InferenceEngine::Blob::Ptr src2,
              InferenceEngine::Blob::Ptr dst)
{
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(src2, nullptr);
    ASSERT_NE(dst,  nullptr);
    int32_t IW1 = 0;
    int32_t IH1 = 0;
    int32_t IC1 = 0;
    int32_t IW2 = 0;
    int32_t IH2 = 0;
    int32_t IC2 = 0;
    int32_t OW = 0;
    int32_t OH = 0;
    int32_t OC = 0;

    get_dims(src1, IW1, IH1, IC1);
    get_dims(src2, IW2, IH2, IC2);
    get_dims(dst, OW, OH, OC);
    
    ASSERT_EQ(std::max(IW1, IW2), OW);
    ASSERT_EQ(std::max(IH1, IH2), OH);
    ASSERT_EQ(std::max(IC1, IC2), OC);

    int32_t IW_max = std::max(IW1, IW2);
    int32_t IH_max = std::max(IH1, IH2);
    int32_t IC_max = std::max(IC1, IC2);

    const uint16_t *src1_data = src1->buffer();
    const uint16_t *src2_data = src2->buffer();
    uint16_t *dst_data = dst->buffer();

    ASSERT_NE(src1_data, nullptr);
    ASSERT_NE(src2_data, nullptr);
    ASSERT_NE(dst_data,  nullptr);

    for (uint32_t w = 0; w < IW_max; w++) {
        for (uint32_t h = 0; h < IH_max; h++) {
            for (uint32_t c = 0; c < IC_max; c++) {
                size_t iidx1 = (c % IC1) + (w % IW1) * IC1 + (h % IH1) * IC1 * IW1;
                size_t iidx2 = (c % IC2) + (w % IW2) * IC2 + (h % IH2) * IC2 * IW2;
                size_t oodx  = c + w * OC  + h *  OC * OW;
                float s1 = PrecisionUtils::f16tof32(src1_data[iidx1]);
                float s2 = PrecisionUtils::f16tof32(src2_data[iidx2]);
                dst_data[oodx] = PrecisionUtils::f32tof16(s1 + s2);
            }
        }
    }
}

class myriadLayersTestsBias_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, int32_t>> {
};

TEST_P(myriadLayersTestsBias_nightly, TestsBias)
{
    auto input_dim = std::get<0>(GetParam());
    auto axis      = std::get<1>(GetParam());
    InferenceEngine::SizeVector input_dim1;
    int32_t pos = axis;
    for (; pos < input_dim.size(); ++pos) {
        input_dim1.push_back(input_dim[pos]);
    }
    SetInputTensors({input_dim, input_dim1});
    SetOutputTensors({input_dim});
    NetworkInit("Bias",
                nullptr,
                0,
                0,
                0,
                InferenceEngine::Precision::FP16);

    ASSERT_TRUE(Infer());
    ASSERT_EQ(_inputMap.size(), 2);
    ASSERT_EQ(_outputMap.size(), 1);
    auto iter = _inputMap.begin();
    auto first_input = iter->second;
    ++iter;
    auto second_input = iter->second;
    ref_bias(first_input, second_input, _refBlob);
    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<InferenceEngine::SizeVector> s_biasDims = {
    {{1, 8, 4, 4}}
};

static std::vector<int32_t> s_biasAxis = {
    0, 1, // only limited axis range is supported for now.
};

static std::vector<InferenceEngine::SizeVector> s_biasDims3D = {
    {{32, 8, 16}}
};

static std::vector<int32_t> s_biasAxis3D = {
    0 // only limited axis range is supported for now.
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsBias_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_biasDims),
        ::testing::ValuesIn(s_biasAxis))
);

INSTANTIATE_TEST_CASE_P(accuracy_3D, myriadLayersTestsBias_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_biasDims3D),
        ::testing::ValuesIn(s_biasAxis3D))
);
