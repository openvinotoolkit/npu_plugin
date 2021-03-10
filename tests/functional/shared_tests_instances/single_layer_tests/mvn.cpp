// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbMvnLayerTest : public MvnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvnLayerTest, basicTest) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

// Accuracy problem for 3-dim input tensor when acrossChannels = false
// Traced in Jira CVS-48579
const std::vector<std::vector<size_t>> inputShapes3D = {
    {1, 32, 17},
    {1, 37, 9}
};

const std::vector<bool> acrossChannels3D = {
    true
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 16, 5, 8},
#if 0
// Test fails for accuracy when Number of channels > 1 (tracked in Jira CVS-48579)
    {3, 32, 17},
    {3, 37, 9},
// Batch size > 1 is not supported by Soft and Custom Layer MVN implementation
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
// Currently input dim > 4 is not supported by KMB-plugin and mcmCompiler
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
#endif
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

INSTANTIATE_TEST_CASE_P(
    smoke_TestsMVN_3D, KmbMvnLayerTest, ::testing::Combine(
        ::testing::ValuesIn(inputShapes3D),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(acrossChannels3D),
        ::testing::ValuesIn(normalizeVariance),
        ::testing::ValuesIn(epsilon),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    ), KmbMvnLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_TestsMVN, KmbMvnLayerTest, ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(acrossChannels),
        ::testing::ValuesIn(normalizeVariance),
        ::testing::ValuesIn(epsilon),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    ), KmbMvnLayerTest::getTestCaseName);