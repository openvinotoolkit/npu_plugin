// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, uint32_t>> myriadLayersTestsSplit_nightly;

TEST_P(myriadLayersTestsSplit_nightly, Split) {
    auto dims = std::get<0>(GetParam());
    auto outputNum = std::get<1>(GetParam());
    IN_OUT_desc input;
    input.push_back(dims);
    SetInputTensors(input);
    IN_OUT_desc output;
    for (size_t i = 0; i < outputNum; ++i) {
        output.push_back(dims);
    }
    SetOutputTensors(output);
    ASSERT_NO_FATAL_FAILURE(
        NetworkInit("Split",
                    nullptr,
                    0,
                    0, 
                    nullptr,
                    InferenceEngine::Precision::FP16, // output precision
                    InferenceEngine::Precision::FP16  // input precision
                    )
    );
    ASSERT_TRUE(Infer());
    auto src = _inputMap.begin()->second;
    auto src_data = src->buffer().as<const uint16_t*>();
    ASSERT_EQ(_outputMap.size(), outputNum);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    int32_t OFFSET[3] = {0};
    for (auto outElem : _outputMap) {
        int32_t OW = 1;
        int32_t OH = 1;
        int32_t OC = 1;
        get_dims(outElem.second, OW, OH, OC);
        ASSERT_EQ(src->size(), outElem.second->size());
        ASSERT_EQ(IW, OW);
        ASSERT_EQ(IH, OH);
        ASSERT_EQ(IC, OC);
        auto dst_data = outElem.second->buffer().as<uint16_t*>();
        for (int32_t ii = 0; ii < OH * OW * OC; ++ii) {
            ASSERT_EQ(dst_data[ii], src_data[ii]) << "actual data : " << PrecisionUtils::f16tof32(dst_data[ii]) << " reference data " << PrecisionUtils::f16tof32(src_data[ii]);
        }
    }
}

static std::vector<InferenceEngine::SizeVector> s_inputSplit4D = {
    {{1, 6, 4, 8}}
};

static std::vector<uint32_t> s_dimensions = {
    1, 2, 3, 4, 5
};

static std::vector<InferenceEngine::SizeVector> s_inputSplit2D = {
    {{1, 4, 8}}
};

static std::vector<InferenceEngine::SizeVector> s_inputSplit1D = {
    {{1, 8}}
};

INSTANTIATE_TEST_CASE_P(accuracy_4D, myriadLayersTestsSplit_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSplit4D),
        ::testing::ValuesIn(s_dimensions))
);

INSTANTIATE_TEST_CASE_P(accuracy_2D, myriadLayersTestsSplit_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSplit2D),
        ::testing::ValuesIn(s_dimensions))
);

INSTANTIATE_TEST_CASE_P(accuracy_1D, myriadLayersTestsSplit_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSplit1D),
        ::testing::ValuesIn(s_dimensions))
);
