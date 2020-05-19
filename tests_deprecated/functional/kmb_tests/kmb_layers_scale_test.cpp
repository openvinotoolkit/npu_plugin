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
#include <memory>

#include "kmb_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

typedef kmbLayerTestBaseWithParam<Dims> kmbLayersTestsScaleParams;

#ifdef ENABLE_MCM_COMPILER
TEST_P(kmbLayersTestsScaleParams, TestsScale) {
    auto param = GetParam();
    tensor_test_params tensor = param;
    std::size_t weightsSize = tensor.n * tensor.c * sizeof(uint16_t);
    std::size_t biasesSize = 0;

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightsSize + biasesSize));

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    _config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);

    SetInputTensor(tensor);
    SetOutputTensor(tensor);
    NetworkInit("ScaleShift", &params, weightsSize, biasesSize, weightsBlob,
        Precision::FP16,  // output precision
        Precision::U8     // input precision
    );
}

const static std::vector<Dims> scaleTensors = {
    {{1, 3, 224, 224}},
    {{1, 6, 112, 112}},
};

INSTANTIATE_TEST_CASE_P(accuracy, kmbLayersTestsScaleParams, ::testing::ValuesIn(scaleTensors));
#endif
