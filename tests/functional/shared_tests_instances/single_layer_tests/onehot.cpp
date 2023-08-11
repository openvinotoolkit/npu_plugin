//
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "kmb_layer_test.hpp"
#include "single_layer_tests/one_hot.hpp"

namespace LayerTestsDefinitions {

class VPUXOneHotLayerTest_VPU3720 : public OneHotLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(VPUXOneHotLayerTest_VPU3720, HW) {
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

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_VPU3720_FP16, VPUXOneHotLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64),
                                            ::testing::ValuesIn(depthVal),
                                            ::testing::Values(ngraph::element::Type_t::f16), ::testing::ValuesIn(onVal),
                                            ::testing::ValuesIn(offVal), ::testing::ValuesIn(axis),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_VPU3720_FP32, VPUXOneHotLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64),
                                            ::testing::ValuesIn(depthVal),
                                            ::testing::Values(ngraph::element::Type_t::f32), ::testing::ValuesIn(onVal),
                                            ::testing::ValuesIn(offVal), ::testing::ValuesIn(axis),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_VPU3720_I32, VPUXOneHotLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64),
                                            ::testing::ValuesIn(depthVal),
                                            ::testing::Values(ngraph::element::Type_t::i32), ::testing::ValuesIn(onVal),
                                            ::testing::ValuesIn(offVal), ::testing::ValuesIn(axis),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_VPU3720_I8, VPUXOneHotLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64),
                                            ::testing::ValuesIn(depthVal),
                                            ::testing::Values(ngraph::element::Type_t::i8), ::testing::ValuesIn(onVal),
                                            ::testing::ValuesIn(offVal), ::testing::ValuesIn(axis),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_VPU3720_U8, VPUXOneHotLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::element::Type_t::i64),
                                            ::testing::ValuesIn(depthVal),
                                            ::testing::Values(ngraph::element::Type_t::u8), ::testing::ValuesIn(onVal),
                                            ::testing::ValuesIn(offVal), ::testing::ValuesIn(axis),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         OneHotLayerTest::getTestCaseName);
}  // namespace
