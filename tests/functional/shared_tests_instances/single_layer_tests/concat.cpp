//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

#include <vector>

namespace LayerTestsDefinitions {

class ConcatLayerTestCommon : public ConcatLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ConcatLayerTest_NPU3700 : public ConcatLayerTestCommon {};
class ConcatLayerTest_NPU3720 : public ConcatLayerTestCommon {};

TEST_P(ConcatLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ConcatLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<int> axes = {0, 1, 2, 3};

std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                         InferenceEngine::Precision::U8};
// Check parameters from InceptionV3
// This test is just attempt to use parameters other than in CPU-plugin.
// Note: NPU-plugin does not support batch-size > 1.
std::vector<int> axes_check = {1};

std::vector<std::vector<std::vector<size_t>>> inShapes_check = {
        {{1, 64, 35, 35}, {1, 64, 35, 35}}, {{1, 64, 35, 35}, {1, 64, 35, 35}, {1, 96, 35, 35}, {1, 32, 35, 35}}};

// ------ NPU3700 ------

INSTANTIATE_TEST_SUITE_P(smoke_Concat, ConcatLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes), ::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ConcatLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Concat_InceptionV3, ConcatLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes_check), ::testing::ValuesIn(inShapes_check),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ConcatLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720 ------

const auto concatParams = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::Values(std::vector<std::vector<size_t>>({{1, 16, 10, 10}, {1, 16, 10, 10}})),
        ::testing::Values(InferenceEngine::Precision::U8), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, ConcatLayerTest_NPU3720, concatParams,
                         ConcatLayerTest_NPU3720::getTestCaseName);

}  // namespace
