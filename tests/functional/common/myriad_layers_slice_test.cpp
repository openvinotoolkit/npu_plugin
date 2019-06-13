// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

typedef myriadLayerTestBaseWithParam<std::tuple<IN_OUT_desc, IN_OUT_desc>> myriadLayersTestsSlice_nightly;

TEST_P(myriadLayersTestsSlice_nightly, Slice) {
    auto dims = std::get<0>(GetParam());
    auto output= std::get<1>(GetParam());

    SetInputTensors(dims);
    SetOutputTensors(output);
    ASSERT_NO_FATAL_FAILURE(
        NetworkInit("Slice",
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
    ASSERT_EQ(_outputMap.size(), output.size());
    auto dim = output.begin();
    int32_t INP[3] = { 0 };
    int32_t i_W = 0;
    int32_t i_H = 0;
    int32_t i_C = 0;
    get_dims(src, INP[0], INP[1], INP[2]);
    /* axis search emulation */
    /* due to IE specific: Slice is converted to split layer and axis */
    /* is not known during the parsing                                */
    int32_t axis;
    for (axis = 0; axis < 3; ++axis) {
        int32_t sum = 0;
        for (auto outElem : _outputMap) {
            int32_t OUTP[3] = {0};
            get_dims(outElem.second, OUTP[0], OUTP[1], OUTP[2]);
            sum += OUTP[axis];
        }
        if (sum ==  INP[axis])
            break;
    }
    int32_t OFFSET[3] = {0};
    for (auto outElem : _outputMap) {
        /* dimensions check */
        auto val1 = dim->rbegin();
        for (auto val2 : outElem.second->dims()) {
            ASSERT_EQ(*val1, val2);
            val1++;
        }
        dim++;
        int32_t OUTP[3] = {0};
        get_dims(outElem.second, OUTP[0], OUTP[1], OUTP[2]);
        auto dst_data = outElem.second->buffer().as<uint16_t*>();
        for (int32_t h = 0; h < OUTP[1]; ++h) {
            for (int32_t w = 0; w < OUTP[0]; ++w) {
                for (int32_t c = 0; c < OUTP[2]; ++c) {
                    size_t iidx = c + OFFSET[2] + INP[2] * ( (w + OFFSET[0])  + (h + OFFSET[1]) * INP[0] );
                    size_t oodx = c + OUTP[2] * ( w  + h * OUTP[0] );
                    ASSERT_EQ(dst_data[oodx], src_data[iidx]) << "actual data : " << PrecisionUtils::f16tof32(dst_data[oodx]) << " reference data " << PrecisionUtils::f16tof32(src_data[iidx]);
                }
            }
        }
        OFFSET[axis] += OUTP[axis];
    }
}

static  std::vector<IN_OUT_desc> s_inputSlice4D = {
      {{{1, 6, 4, 8}}}
};

static  std::vector<IN_OUT_desc> s_slice4D = {
    {{{1, 1, 4, 8}, {1, 2, 4, 8}, {1, 3, 4, 8}}},
    {{{1, 6, 4, 1}, {1, 6, 4, 2}, {1, 6, 4, 1}, {1, 6, 4, 4}}},
    {{{1, 6, 1, 8}, {1, 6, 2, 8}, {1, 6, 1, 8}}}
};

static  std::vector<IN_OUT_desc> s_inputSlice1D = {
      {{{1, 8}}}
};

static  std::vector<IN_OUT_desc> s_slice1D = {
    {{{1, 2}, {1, 4}, {1, 2}}},
};

static  std::vector<IN_OUT_desc> s_inputSlice2D = {
      {{{1, 4, 8}}}
};

static  std::vector<IN_OUT_desc> s_slice2D = {
    {{{1, 4, 2}, {1, 4, 4}, {1, 4, 2}}},
    {{{1, 1, 8}, {1, 2, 8}, {1, 1, 8}}}
};

INSTANTIATE_TEST_CASE_P(accuracy_4D, myriadLayersTestsSlice_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSlice4D),
        ::testing::ValuesIn(s_slice4D))
);

INSTANTIATE_TEST_CASE_P(accuracy_2D, myriadLayersTestsSlice_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSlice2D),
        ::testing::ValuesIn(s_slice2D))
);

INSTANTIATE_TEST_CASE_P(accuracy_1D, myriadLayersTestsSlice_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_inputSlice1D),
        ::testing::ValuesIn(s_slice1D))
);