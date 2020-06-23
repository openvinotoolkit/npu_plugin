//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

struct clamp_test_params {
    float min;
    float max;
    friend std::ostream& operator<<(std::ostream& os, clamp_test_params const& tst) {
        return os << " min=" << tst.min << ", max=" << tst.max;
    };
};

typedef kmbLayerTestBaseWithParam<std::tuple<Dims, clamp_test_params>> kmbLayersTestsClampParams_nightly;

#ifdef ENABLE_MCM_COMPILER

// [Track number: S#27230]
TEST_P(kmbLayersTestsClampParams_nightly, DISABLED_TestsClamp) {
    auto param = GetParam();
    tensor_test_params tensor = std::get<0>(param);
    clamp_test_params p = std::get<1>(param);

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;
    params["min"] = std::to_string(p.min);
    params["max"] = std::to_string(p.max);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("Clamp", &params, 0, 0, nullptr,
        InferenceEngine::Precision::FP16  // output precision
    );
    /* input data preparation */
    //    SetFirstInputToRange(-100.f, 100.f);
    //    ASSERT_TRUE(Infer());
    //
    //    /* output check */
    //    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    //    auto inputBlob  = _inputMap[_inputsInfo.begin()->first];

    //    ref_Clamp(inputBlob, _refBlob, p.min, p.max);
    //
    //    Compare(outputBlob, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_clampTensors = {
    {{1, 3, 10, 15}},
    {{1, 11, 10, 15}},
};

static std::vector<clamp_test_params> s_clampParams = {
    {0.f, 6.0f}, {1.f, 3.0f},
    //    {-10.f, 17.0f}
};

INSTANTIATE_TEST_CASE_P(accuracy, kmbLayersTestsClampParams_nightly,
    ::testing::Combine(::testing::ValuesIn(s_clampTensors), ::testing::ValuesIn(s_clampParams)));
#endif
