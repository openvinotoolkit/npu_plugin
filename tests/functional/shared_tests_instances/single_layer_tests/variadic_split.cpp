//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/single_layer/variadic_split.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXVariadicSplitLayerTest :
        public VariadicSplitLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

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

const auto variadicSplitParams0 =
        testing::Combine(::testing::Values(numSplits),                                // numSplits
                         ::testing::Values(1),                                        // axis
                         ::testing::Values(netPrecisions),                            // netPrecision
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::ValuesIn(inputShapes),                            // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams1 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams2 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams3 =
        testing::Combine(::testing::Values(std::vector<size_t>{2, 4, 4}),                 // numSplits
                         ::testing::Values(0, 1, 2, 3),                                   // axis
                         ::testing::Values(netPrecisions),                                // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),                 // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),                 // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{10, 10, 10, 10}),  // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams4 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 4, 2}),     // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// Tracking number [E#85137]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_VariadicSplit, VPUXVariadicSplitLayerTest_VPU3700, variadicSplitParams0,
                        VPUXVariadicSplitLayerTest_VPU3700::getTestCaseName);

/* ============= Negative Axis ============= */

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis, VPUXVariadicSplitLayerTest_VPU3700, variadicSplitParams1,
                        VPUXVariadicSplitLayerTest_VPU3700::getTestCaseName);

/* ============= VPU3720  ============= */
INSTANTIATE_TEST_CASE_P(smoke_VariadicSplit, VPUXVariadicSplitLayerTest_VPU3720, variadicSplitParams0,
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis0, VPUXVariadicSplitLayerTest_VPU3720, variadicSplitParams1,
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis1, VPUXVariadicSplitLayerTest_VPU3720, variadicSplitParams2,
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitPosAxis, VPUXVariadicSplitLayerTest_VPU3720, variadicSplitParams3,
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_VariadicSplitNegAxis, VPUXVariadicSplitLayerTest_VPU3720, variadicSplitParams4,
                        VPUXVariadicSplitLayerTest_VPU3720::getTestCaseName);

}  // namespace
