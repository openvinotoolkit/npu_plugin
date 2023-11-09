//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/minimum_maximum.hpp"
#include <common/functions.h>
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXMaxMinLayerTest : public MaxMinLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class VPUXMaxMinLayerTest_VPU3700 : public VPUXMaxMinLayerTest {};
class VPUXMaxMinLayerTest_VPU3720 : public VPUXMaxMinLayerTest {};

TEST_P(VPUXMaxMinLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMaxMinLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<std::vector<size_t>>> inShapes4D = {{{1, 64, 32, 32}, {1, 64, 32, 32}},
                                                                  {{1, 1, 1, 3}, {1}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

const std::vector<ngraph::helpers::MinMaxOpType> opType = {
        ngraph::helpers::MinMaxOpType::MINIMUM,
        ngraph::helpers::MinMaxOpType::MAXIMUM,
};

const std::vector<ngraph::helpers::InputLayerType> inputType = {ngraph::helpers::InputLayerType::CONSTANT};

// Tracking number [E#85137]
const std::vector<InferenceEngine::Layout> layout4D = {InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC};

const std::vector<std::vector<std::vector<size_t>>> inShapes3D = {{{1, 2, 4}, {1}}};
const auto params0 =
        testing::Combine(::testing::ValuesIn(inShapes4D), ::testing::ValuesIn(opType),
                         ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::ValuesIn(layout4D),
                         ::testing::ValuesIn(layout4D), ::testing::ValuesIn(inputType),
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto params1 = testing::Combine(
        ::testing::ValuesIn(inShapes3D), ::testing::ValuesIn(opType), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(inputType),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// VPU3700 Instantiation
//
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_maximum_4D, VPUXMaxMinLayerTest_VPU3700, params0,
                         VPUXMaxMinLayerTest_VPU3700::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_maximum_3D, VPUXMaxMinLayerTest_VPU3700, params1,
                         VPUXMaxMinLayerTest_VPU3700::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapesVPU3720 = {{{1, 1, 16, 32}, {1, 1, 16, 32}}, {{32}, {1}}};

const auto params2 = testing::Combine(
        ::testing::ValuesIn(inShapesVPU3720), ::testing::ValuesIn(opType), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(inputType),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params3 = testing::Combine(
        ::testing::Values(std::vector<std::vector<size_t>>({{1, 1, 1, 3}, {1}})), ::testing::ValuesIn(opType),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(inputType),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// VPU3720 Instantiation
//
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_VPU3720_test0, VPUXMaxMinLayerTest_VPU3720, params0,
                         VPUXMaxMinLayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_VPU3720_test1, VPUXMaxMinLayerTest_VPU3720, params1,
                         VPUXMaxMinLayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_VPU3720_test2, VPUXMaxMinLayerTest_VPU3720, params2,
                         VPUXMaxMinLayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_VPU3720_test3, VPUXMaxMinLayerTest_VPU3720, params3,
                         VPUXMaxMinLayerTest_VPU3720::getTestCaseName);

}  // namespace
