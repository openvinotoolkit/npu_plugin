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

typedef kmbLayerTestBaseWithParam<std::tuple<Dims, size_t>> kmbLayersTestsSoftMaxParams;

#ifdef ENABLE_MCM_COMPILER
TEST_P(kmbLayersTestsSoftMaxParams, TestsSoftMax) {
    auto param = GetParam();
    tensor_test_params tensor = std::get<0>(param);
    size_t axis = std::get<1>(param);

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;
    params["axis"] = std::to_string(axis);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    _config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("SoftMax", &params, 0, 0, nullptr,
        InferenceEngine::Precision::FP16  // output precision
    );
}

static std::vector<Dims> s_softMaxTensors = {
    {{1, 1001, 1, 1}},
    {{1, 1000, 1, 1}},
};

INSTANTIATE_TEST_CASE_P(accuracy, kmbLayersTestsSoftMaxParams,
    ::testing::Combine(::testing::ValuesIn(s_softMaxTensors), ::testing::Values(1, 2)));
#endif
