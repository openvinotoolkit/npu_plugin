//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/minimum_maximum.hpp"
#include <common/functions.h>
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class MaxMinLayerTestCommon : public MaxMinLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class MaxMinLayerTest_NPU3700 : public MaxMinLayerTestCommon {};
class MaxMinLayerTest_NPU3720 : public MaxMinLayerTestCommon {};

TEST_P(MaxMinLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(MaxMinLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
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
const std::vector<std::vector<std::vector<size_t>>> inShapes4D = {{{1, 64, 32, 32}, {1, 64, 32, 32}},
                                                                  {{1, 1, 1, 3}, {1}}};
const std::vector<std::vector<std::vector<size_t>>> inShapesGeneric = {{{1, 1, 16, 32}, {1, 1, 16, 32}}, {{32}, {1}}};

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

const auto params2 = testing::Combine(
        ::testing::ValuesIn(inShapesGeneric), ::testing::ValuesIn(opType), ::testing::ValuesIn(netPrecisions),
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
// NPU3700 Instantiation
//
INSTANTIATE_TEST_SUITE_P(smoke_maximum_4D, MaxMinLayerTest_NPU3700, params0, MaxMinLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_maximum_3D, MaxMinLayerTest_NPU3700, params1, MaxMinLayerTest_NPU3700::getTestCaseName);

//
// NPU3720 Instantiation
//
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test0, MaxMinLayerTest_NPU3720, params0,
                         MaxMinLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test1, MaxMinLayerTest_NPU3720, params1,
                         MaxMinLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test2, MaxMinLayerTest_NPU3720, params2,
                         MaxMinLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test3, MaxMinLayerTest_NPU3720, params3,
                         MaxMinLayerTest_NPU3720::getTestCaseName);

}  // namespace
