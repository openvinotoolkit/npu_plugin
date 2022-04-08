// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roll.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbRollLayerTest : public RollLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbRollLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::U8, InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,  // CPU-plugin has parameter I16, but KMB does not support it. So value from
                                           // CPU-plugin I16 is changed to FP16.
        InferenceEngine::Precision::FP32};

std::vector<std::vector<size_t>> inputShapes = {
        {16},            // testCase1D
        {17, 19},        // testCase2DZeroShifts
        {600, 450},      // testCase2D
        {2, 320, 320},   // testCase3D
        {3, 11, 6, 4},   // testCaseNegativeUnorderedAxes4D
        {2, 16, 32, 32}  // testCaseRepeatingAxes5D
};

const std::vector<std::vector<int64_t>> shift = {{5},        {0, 0}, {300, 250},
                                                 {160, 160}, {7, 3}, {16, 15, 10, 2, 1, 7, 2, 8, 1, 1}};

const std::vector<std::vector<int64_t>> axes = {{0},    {0, 1},   {0, 1},
                                                {1, 2}, {-3, -2}, {-1, -2, -3, 1, 0, 3, 3, 2, -2, -3}};

const auto testRollParams0 = ::testing::Combine(::testing::Values(inputShapes[0]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[0]), ::testing::Values(axes[0]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check0, KmbRollLayerTest, testRollParams0, KmbRollLayerTest::getTestCaseName);

const auto testRollParams1 = ::testing::Combine(::testing::Values(inputShapes[1]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[1]), ::testing::Values(axes[1]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check1, KmbRollLayerTest, testRollParams1, KmbRollLayerTest::getTestCaseName);

const auto testRollParams2 = ::testing::Combine(::testing::Values(inputShapes[2]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[2]), ::testing::Values(axes[2]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check2, KmbRollLayerTest, testRollParams2, KmbRollLayerTest::getTestCaseName);

const auto testRollParams3 = ::testing::Combine(::testing::Values(inputShapes[3]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[3]), ::testing::Values(axes[3]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check3, KmbRollLayerTest, testRollParams3, KmbRollLayerTest::getTestCaseName);

const auto testRollParams4 = ::testing::Combine(::testing::Values(inputShapes[4]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[4]), ::testing::Values(axes[4]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check4, KmbRollLayerTest, testRollParams4, KmbRollLayerTest::getTestCaseName);

const auto testRollParams5 = ::testing::Combine(::testing::Values(inputShapes[5]), ::testing::ValuesIn(inputPrecisions),
                                                ::testing::Values(shift[5]), ::testing::Values(axes[5]),
                                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Roll_Test_Check5, KmbRollLayerTest, testRollParams5, KmbRollLayerTest::getTestCaseName);

}  // namespace
