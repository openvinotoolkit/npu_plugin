//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/transpose.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXTransposeLayerTest : public TransposeLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class VPUXTransposeLayerTest_VPU3700 : public VPUXTransposeLayerTest {};
class VPUXTransposeLayerTest_VPU3720 : public VPUXTransposeLayerTest {};

TEST_P(VPUXTransposeLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXTransposeLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

// MLIR 2D instantiation

const std::vector<std::vector<size_t>> inputShapes2D = {
        std::vector<size_t>{50, 100},
};

const std::vector<std::vector<size_t>> inputOrder2D = {
        std::vector<size_t>{},
};

const auto params2D = testing::Combine(testing::ValuesIn(inputOrder2D), testing::ValuesIn(netPrecisions),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Layout::ANY),
                                       testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapes2D),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// [Track number: W#7312]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Transpose2D, VPUXTransposeLayerTest_VPU3700, params2D,
                         VPUXTransposeLayerTest_VPU3700::getTestCaseName);

// MLIR 4D instantiation

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{1, 3, 100, 100},
};

// Tracking number [E#85137]
const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 3, 2, 1},
};

const auto params4D = testing::Combine(testing::ValuesIn(inputOrder4D), testing::ValuesIn(netPrecisions),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Layout::ANY),
                                       testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapes4D),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Transpose4D, VPUXTransposeLayerTest_VPU3700, params4D,
                         VPUXTransposeLayerTest_VPU3700::getTestCaseName);

// MLIR 4D MemPermute instantiation

const std::vector<std::vector<size_t>> inputShapesMemPerm = {
        std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrderMemPerm = {
        std::vector<size_t>{0, 2, 3, 1},
};

const auto paramsMemPermNCHWtoNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::NCHW), testing::Values(InferenceEngine::Layout::NHWC),
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNCHW, VPUXTransposeLayerTest_VPU3700, paramsMemPermNCHWtoNHWC,
                        VPUXTransposeLayerTest_VPU3700::getTestCaseName);

const auto paramsMemPermInNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::NHWC),
        testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapesMemPerm),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNHWC, VPUXTransposeLayerTest_VPU3700, paramsMemPermInNHWC,
                        VPUXTransposeLayerTest_VPU3700::getTestCaseName);

// ------ VPU3720 ------

const std::vector<std::vector<size_t>> inputOrderVPU3720 = {
        std::vector<size_t>{0, 2, 3, 1},
};

const auto paramsVPU3720 = testing::Combine(
        testing::ValuesIn(inputOrderVPU3720), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TransposeVPU3720, VPUXTransposeLayerTest_VPU3720, paramsVPU3720,
                        VPUXTransposeLayerTest_VPU3720::getTestCaseName);

// -------- ND ---------

const std::vector<std::vector<size_t>> shape_5D = {std::vector<size_t>{1, 10, 10, 4, 6},
                                                   std::vector<size_t>{1, 10, 4, 6, 1}};
const std::vector<std::vector<size_t>> reorder_5D = {std::vector<size_t>{4, 1, 2, 3, 0},
                                                     std::vector<size_t>{4, 0, 2, 3, 1}};

const auto paramsVPU3720_5D = testing::Combine(
        testing::ValuesIn(reorder_5D), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(shape_5D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_TransposeVPU3720_5D, VPUXTransposeLayerTest_VPU3720, paramsVPU3720_5D,
                        VPUXTransposeLayerTest_VPU3720::getTestCaseName);

}  // namespace
