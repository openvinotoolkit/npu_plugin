//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

#include <vector>

namespace LayerTestsDefinitions {
class KmbConcatLayerTest : public ConcatLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    // [Track number: S#49997]
    virtual void SkipBeforeLoad() override {
    }
    // [Track number: S#49998]
    virtual void SkipBeforeInfer() override {
    }
};

class KmbConcatLayerTest_VPU3720 : public ConcatLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConcatLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbConcatLayerTest, CompareWithRefs_SW_MLIR) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbConcatLayerTest_VPU3720, CompareWithRefs_SW_MLIR_VPU3720) {
    useCompilerMLIR();
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

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Concat, KmbConcatLayerTest,
                         ::testing::Combine(::testing::ValuesIn(axes), ::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConcatLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, KmbConcatLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(axes),
                                            ::testing::Values(std::vector<std::vector<size_t>>({{1, 16, 10, 10},
                                                                                                {1, 16, 10, 10}})),
                                            ::testing::Values(InferenceEngine::Precision::U8),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConcatLayerTest_VPU3720::getTestCaseName);

// Check parameters from InceptionV3
// This test is just attempt to use parameters other than in CPU-plugin.
// Note: VPUx-plugin does not support batch-size > 1.
std::vector<int> axes_check = {1};

std::vector<std::vector<std::vector<size_t>>> inShapes_check = {
        {{1, 64, 35, 35}, {1, 64, 35, 35}}, {{1, 64, 35, 35}, {1, 64, 35, 35}, {1, 96, 35, 35}, {1, 32, 35, 35}}};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Concat_InceptionV3, KmbConcatLayerTest,
                         ::testing::Combine(::testing::ValuesIn(axes_check), ::testing::ValuesIn(inShapes_check),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConcatLayerTest::getTestCaseName);

}  // namespace
