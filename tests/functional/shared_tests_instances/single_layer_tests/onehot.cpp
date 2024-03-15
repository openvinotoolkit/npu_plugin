//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include "single_layer_tests/one_hot.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class OneHotLayerTestCommon : public OneHotLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class OneHotLayerTest_NPU3720 : public OneHotLayerTestCommon {};

TEST_P(OneHotLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<int64_t> depthVal{3};
const std::vector<float> onVal{1.0f};
const std::vector<float> offVal{0.0f};
const std::vector<int64_t> axis{-2, 0};
const std::vector<InferenceEngine::SizeVector> inputShape = {
        InferenceEngine::SizeVector{4},
        InferenceEngine::SizeVector{2, 3},
};

auto oneHotparams = [](auto onOffType) {
    return ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64), ::testing::ValuesIn(depthVal),
                              ::testing::Values(onOffType), ::testing::ValuesIn(onVal), ::testing::ValuesIn(offVal),
                              ::testing::ValuesIn(axis), ::testing::Values(InferenceEngine::Precision::I32),
                              ::testing::ValuesIn(inputShape),
                              ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
};

// ------ NPU3720 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_OneHot_FP16, OneHotLayerTest_NPU3720,
                         oneHotparams(ngraph::element::Type_t::f16), OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_FP32, OneHotLayerTest_NPU3720, oneHotparams(ngraph::element::Type_t::f32),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I32, OneHotLayerTest_NPU3720, oneHotparams(ngraph::element::Type_t::i32),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I8, OneHotLayerTest_NPU3720, oneHotparams(ngraph::element::Type_t::i8),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_U8, OneHotLayerTest_NPU3720, oneHotparams(ngraph::element::Type_t::u8),
                         OneHotLayerTest::getTestCaseName);

}  // namespace
