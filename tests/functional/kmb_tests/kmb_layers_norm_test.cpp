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

typedef std::tuple<tensor_test_params, float, float, size_t, std::string> norm_test_params;
typedef kmbLayerTestBaseWithParam<norm_test_params> kmbLayersTestsNormParams;

#ifdef ENABLE_MCM_COMPILER
TEST_P(kmbLayersTestsNormParams, DISABLED_TestsNorm) {
    auto param = GetParam();
    tensor_test_params tensor = std::get<0>(param);
    float alpha = std::get<1>(param);
    float beta = std::get<2>(param);
    unsigned long localSize = std::get<3>(param);
    std::string region = std::get<4>(param);

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;

    params["alpha"] = std::to_string(alpha);
    params["beta"] = std::to_string(beta);
    params["local-size"] = std::to_string(localSize);
    params["region"] = region;

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    _config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("Norm", &params, 0, 0, nullptr,
        InferenceEngine::Precision::FP16  // output precision
    );
}

static const norm_test_params paramsTable[] = {
    std::make_tuple<tensor_test_params, float, float, size_t, std::string>({1, 96, 6, 6},  // input and output tensors
        9.9999997e-05,                                                                     // alpha
        0.75,                                                                              // beta
        5,                                                                                 // local-size
        "across"                                                                           // region
        ),
};

INSTANTIATE_TEST_CASE_P(loadNetworkNoThrow, kmbLayersTestsNormParams, ::testing::ValuesIn(paramsTable));
#endif
