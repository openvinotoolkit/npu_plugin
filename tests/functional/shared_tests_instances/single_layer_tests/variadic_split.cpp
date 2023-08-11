// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/variadic_split.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXVariadicSplitLayerTest :
        public VariadicSplitLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXVariadicSplitLayerTest_VPU3700 : public VPUXVariadicSplitLayerTest {};
class VPUXVariadicSplitLayerTest_VPU3720 : public VPUXVariadicSplitLayerTest {};

TEST_P(VPUXVariadicSplitLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
TEST_P(VPUXVariadicSplitLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::SizeVector> inputShapes = {InferenceEngine::SizeVector{1, 144, 30, 40}};

const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP32;

const std::vector<size_t> numSplits = {64, 48, 32};

// [Track number: E#28335]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_VariadicSplit, VPUXVariadicSplitLayerTest_VPU3700,
                        ::testing::Combine(::testing::Values(numSplits),                                // numSplits
                                           ::testing::Values(1),                                        // axis
                                           ::testing::Values(netPrecisions),                            // netPrecision
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                                           ::testing::ValuesIn(inputShapes),                            // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXVariadicSplitLayerTest_VPU3700::getTestCaseName);

/* ============= Negative Axis ============= */

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis, VPUXVariadicSplitLayerTest_VPU3700,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                                           ::testing::Values(-1),                                       // axis
                                           ::testing::Values(netPrecisions),                            // netPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                                           ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXVariadicSplitLayerTest_VPU3700::getTestCaseName);

/* ============= VPU3720  ============= */

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis, VPUXVariadicSplitLayerTest_VPU3720,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                                           ::testing::Values(-1),                                       // axis
                                           ::testing::Values(netPrecisions),                            // netPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                                           ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitPosAxis, VPUXVariadicSplitLayerTest_VPU3720,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{2, 4, 4}),             // numSplits
                                           ::testing::Values(0, 1, 2, 3),                               // axis
                                           ::testing::Values(netPrecisions),                            // netPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                                           ::testing::Values(InferenceEngine::SizeVector{10, 10, 10,
                                                                                         10}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_VariadicSplitNegAxis, VPUXVariadicSplitLayerTest_VPU3720,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                                           ::testing::Values(-1),                                       // axis
                                           ::testing::Values(netPrecisions),                            // netPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                                           ::testing::Values(InferenceEngine::SizeVector{1, 4, 2}),     // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);
}  // namespace
