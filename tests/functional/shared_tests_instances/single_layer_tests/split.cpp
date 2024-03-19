//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/single_layer/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class SplitLayerTestCommon : public SplitLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class SplitLayerTest_NPU3700 : public SplitLayerTestCommon {
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::VpuSkipTestException(
                "Issues with Runtime. Outputs is empty because runtime doesn't wait while dma is finished");
    }
};

class SplitLayerTest_NPU3720 : public SplitLayerTestCommon {};

TEST_P(SplitLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(SplitLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        InferenceEngine::Precision::FP16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const auto paramsConfig1 =
        testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1, 2, 3), ::testing::ValuesIn(netPrecisions),
                         ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                         ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                         ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                         ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                         ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                         ::testing::Values(InferenceEngine::SizeVector({})),
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto paramsConfig2 =
        testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3), ::testing::ValuesIn(netPrecisions),
                         ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                         ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                         ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                         ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                         ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                         ::testing::Values(InferenceEngine::SizeVector({})),
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto paramsPrecommit = testing::Combine(
        ::testing::Values(2, 3), ::testing::Values(0), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
        ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
        ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
        ::testing::Values(InferenceEngine::SizeVector({})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// --------- NPU3700 ---------

INSTANTIATE_TEST_SUITE_P(smoke_Split, SplitLayerTest_NPU3700, paramsConfig1, SplitLayerTest::getTestCaseName);

// --------- NPU3720 ---------

INSTANTIATE_TEST_SUITE_P(smoke_Split, SplitLayerTest_NPU3720, paramsConfig1, SplitLayerTest::getTestCaseName);

}  // namespace
