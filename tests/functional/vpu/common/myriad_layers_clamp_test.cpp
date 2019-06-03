// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

struct clamp_test_params {
    float min;
    float max;
    friend std::ostream& operator<<(std::ostream& os, clamp_test_params const& tst)
    {
        return os << " min=" << tst.min
                  << ", max=" << tst.max;
    };
};

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, clamp_test_params>> myriadLayersTestsClampParams_nightly;

TEST_P(myriadLayersTestsClampParams_nightly, TestsClamp) {
    auto param = GetParam();
    tensor_test_params tensor = std::get<0>(param);
    clamp_test_params p = std::get<1>(param);

    std::map<std::string, std::string> params;
    params["min"] = std::to_string(p.min);
    params["max"] = std::to_string(p.max);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("Clamp",
                &params,
                0,
                0,
                nullptr,
                InferenceEngine::Precision::FP16 // output precision
    );
    /* input data preparation */
    SetFirstInputToRange(-100.f, 100.f);
    ASSERT_TRUE(Infer());

    /* output check */
    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    auto inputBlob  = _inputMap[_inputsInfo.begin()->first];

    ref_Clamp(inputBlob, _refBlob, p.min, p.max);

    Compare(outputBlob, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_clampTensors = {
    {{1, 3, 10, 15}},
};

static std::vector<clamp_test_params> s_clampParams = {
    {0.f, 6.0f},
    {-10.f, 17.0f}
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsClampParams_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_clampTensors),
        ::testing::ValuesIn(s_clampParams))
);
