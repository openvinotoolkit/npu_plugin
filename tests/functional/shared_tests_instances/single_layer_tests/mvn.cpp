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

class KmbMvn6LayerTest : public Mvn6LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvn6LayerTest, basicTest) {
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

INSTANTIATE_TEST_SUITE_P(
    smoke_TestsMVN_3D, KmbMvnLayerTest, ::testing::Combine(
        ::testing::ValuesIn(inputShapes3D),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(acrossChannels3D),
        ::testing::ValuesIn(normalizeVariance),
        ::testing::ValuesIn(epsilon),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    ), KmbMvnLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_TestsMVN, KmbMvnLayerTest, ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(acrossChannels),
        ::testing::ValuesIn(normalizeVariance),
        ::testing::ValuesIn(epsilon),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    ), KmbMvnLayerTest::getTestCaseName);

//Test MVN-6

const std::vector<std::string> epsMode = {
    "inside_sqrt",
    "outside_sqrt"
};

const std::vector<float> epsilonF = {
    0.0001
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_4D, KmbMvn6LayerTest, ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 10, 5, 17}}),
                            ::testing::Values(InferenceEngine::Precision::FP16),
                            ::testing::Values(InferenceEngine::Precision::I32),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::Values("outside_sqrt"),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMvn6LayerTest::getTestCaseName);
