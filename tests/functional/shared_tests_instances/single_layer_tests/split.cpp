//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXSplitLayerTest : public SplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXSplitLayerTest_VPU3700 : public VPUXSplitLayerTest {
    void SkipBeforeLoad() override {
    }

    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException(
                "Issues with Runtime. Outputs is empty because runtime doesn't wait while dma is finished");
    }
};
class VPUXSplitLayerTest_VPU3720 : public VPUXSplitLayerTest {};

TEST_P(VPUXSplitLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXSplitLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        InferenceEngine::Precision::FP16   // tests: VPUXGRNLayerTest, VPUXSplitLayerTest, VPUXCTCGreedyDecoderLayerTest
};

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Split, VPUXSplitLayerTest_VPU3700,
        ::testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1, 2, 3), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                           ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                           ::testing::Values(InferenceEngine::SizeVector({})),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Split, VPUXSplitLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(2), ::testing::Values(1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                                            ::testing::Values(InferenceEngine::SizeVector({})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         SplitLayerTest::getTestCaseName);

}  // namespace
