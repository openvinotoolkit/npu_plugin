//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "single_layer_tests/adaptive_pooling.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXAdaPoolLayerTest : public AdaPoolLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class VPUXAdaPoolLayerTest_VPU3700 : public VPUXAdaPoolLayerTest {};
class VPUXAdaPoolLayerTest_VPU3720 :
        public VPUXAdaPoolLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

TEST_P(VPUXAdaPoolLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXAdaPoolLayerTest_VPU3720, SW) {
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

/* ============= 3D/4D/5D AdaptiveAVGPool VPU3700 ============= */

const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}, {1, 1, 3}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),     // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),      // mode
                           ::testing::ValuesIn(netPrecisions),                               // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));  // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool3D, VPUXAdaPoolLayerTest_VPU3700, AdaPool3DCases,
                        VPUXAdaPoolLayerTest_VPU3700::getTestCaseName);

const auto AdaPool4DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 32, 32}, {1, 1, 3, 2}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{3, 5}, {16, 16}}),                 // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                          // mode
        ::testing::ValuesIn(netPrecisions),                                                   // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));                      // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool4D, VPUXAdaPoolLayerTest_VPU3700, AdaPool4DCases,
                        VPUXAdaPoolLayerTest_VPU3700::getTestCaseName);

const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 17, 4, 5, 4}, {1, 1, 3, 2, 3}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {3, 5, 3}}),                  // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                               // mode
        ::testing::ValuesIn(netPrecisions),                                                        // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));                           // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool5D, VPUXAdaPoolLayerTest_VPU3700, AdaPool5DCases,
                        VPUXAdaPoolLayerTest_VPU3700::getTestCaseName);

/* ============= 3D/4D AdaptiveAVGPool VPU3720 ============= */

const auto AdaPoolCase3D =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}}),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool3D, VPUXAdaPoolLayerTest_VPU3720, AdaPoolCase3D,
                        VPUXAdaPoolLayerTest_VPU3720::getTestCaseName);

const auto AdaPoolCase4D =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 128, 32, 64}}),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{2, 2}, {3, 3}, {6, 6}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool4D, VPUXAdaPoolLayerTest_VPU3720, AdaPoolCase4D,
                        VPUXAdaPoolLayerTest_VPU3720::getTestCaseName);
