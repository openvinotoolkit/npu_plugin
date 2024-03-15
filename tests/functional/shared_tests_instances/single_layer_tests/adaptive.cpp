// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "single_layer_tests/adaptive_pooling.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class AdaPoolLayerTestCommon : public AdaPoolLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class AdaPoolLayerTest_NPU3700 : public AdaPoolLayerTestCommon {};
class AdaPoolLayerTest_NPU3720 : public AdaPoolLayerTestCommon {};

TEST_P(AdaPoolLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(AdaPoolLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,

};

/* ============= 3D/4D/5D AdaptiveAVGPool NPU3700 ============= */

const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}, {1, 1, 3}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),     // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),      // mode
                           ::testing::ValuesIn(netPrecisions),                               // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));  // device

const auto AdaPool4DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 32, 32}, {1, 1, 3, 2}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{3, 5}, {16, 16}}),                 // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                          // mode
        ::testing::ValuesIn(netPrecisions),                                                   // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));                      // device

const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 17, 4, 5, 4}, {1, 1, 3, 2, 3}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {3, 5, 3}}),                  // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                               // mode
        ::testing::ValuesIn(netPrecisions),                                                        // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));                           // device

// ------ NPU3700 ------

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool3D, AdaPoolLayerTest_NPU3700, AdaPool3DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool4D, AdaPoolLayerTest_NPU3700, AdaPool4DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool5D, AdaPoolLayerTest_NPU3700, AdaPool5DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

/* ============= 3D/4D AdaptiveAVGPool NPU3720 ============= */

const auto AdaPoolCase3D =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}}),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto AdaPoolCase4D =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 128, 32, 64}}),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{2, 2}, {3, 3}, {6, 6}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool3D, AdaPoolLayerTest_NPU3720, AdaPoolCase3D,
                        AdaPoolLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool4D, AdaPoolLayerTest_NPU3720, AdaPoolCase4D,
                        AdaPoolLayerTest_NPU3720::getTestCaseName);
