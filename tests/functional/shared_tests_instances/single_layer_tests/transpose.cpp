//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/transpose.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class TransposeLayerTestCommon : public TransposeLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class TransposeLayerTest_NPU3700 : public TransposeLayerTestCommon {};
class TransposeLayerTest_NPU3720 : public TransposeLayerTestCommon {};

TEST_P(TransposeLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(TransposeLayerTest_NPU3720, HW) {
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

// MLIR 4D instantiation
const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{1, 3, 100, 100},
};

// Tracking number [E#85137]
const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 3, 2, 1},
};

const std::vector<std::vector<size_t>> inputShapesMemPerm = {
        std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrderMemPerm = {
        std::vector<size_t>{0, 2, 3, 1},
};

/* ============= NPU3700  ============= */

const auto params2D = testing::Combine(testing::ValuesIn(inputOrder2D), testing::ValuesIn(netPrecisions),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Layout::ANY),
                                       testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapes2D),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params4D = testing::Combine(testing::ValuesIn(inputOrder4D), testing::ValuesIn(netPrecisions),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                       testing::Values(InferenceEngine::Layout::ANY),
                                       testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapes4D),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsMemPermNCHWtoNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::NCHW), testing::Values(InferenceEngine::Layout::NHWC),
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsMemPermInNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::NHWC),
        testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapesMemPerm),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// [Track number: W#7312]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Transpose2D, TransposeLayerTest_NPU3700, params2D,
                         TransposeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Transpose4D, TransposeLayerTest_NPU3700, params4D,
                         TransposeLayerTest_NPU3700::getTestCaseName);

// MLIR 4D MemPermute instantiation
INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNCHW, TransposeLayerTest_NPU3700, paramsMemPermNCHWtoNHWC,
                        TransposeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNHWC, TransposeLayerTest_NPU3700, paramsMemPermInNHWC,
                        TransposeLayerTest_NPU3700::getTestCaseName);

/* ============= NPU3720  ============= */

const auto paramsNPU3720 = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_Transpose, TransposeLayerTest_NPU3720, paramsNPU3720,
                        TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 ND  ============= */

const std::vector<std::vector<size_t>> shape_5D = {std::vector<size_t>{1, 10, 10, 4, 6},
                                                   std::vector<size_t>{1, 10, 4, 6, 1}};
const std::vector<std::vector<size_t>> reorder_5D = {std::vector<size_t>{4, 1, 2, 3, 0},
                                                     std::vector<size_t>{4, 0, 2, 3, 1}};

const auto params_5D = testing::Combine(
        testing::ValuesIn(reorder_5D), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(shape_5D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_Transpose_5D, TransposeLayerTest_NPU3720, params_5D,
                        TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test complex tensor optimization  ============= */

const std::vector<std::vector<size_t>> orderCNo4d = {std::vector<size_t>{0, 1, 3, 2, 4},
                                                     std::vector<size_t>{0, 3, 2, 1, 4}};

const std::vector<std::vector<size_t>> inputShapesCNo4d = {
        std::vector<size_t>{1, 3, 8, 8, 2}, std::vector<size_t>{1, 3, 4, 5, 2}, std::vector<size_t>{1, 3, 2, 5, 2},
        std::vector<size_t>{1, 3, 5, 2, 2}, std::vector<size_t>{1, 3, 9, 7, 2}, std::vector<size_t>{1, 2, 33, 33, 2}};

const auto paramsCNo4d = testing::Combine(
        testing::ValuesIn(orderCNo4d), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesCNo4d), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TransposeCNo4d, TransposeLayerTest_NPU3720, paramsCNo4d,
                        TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test optimization with merged input shape 4D ============= */

const std::vector<std::vector<size_t>> orderMerged4d = {
        std::vector<size_t>{0, 2, 1, 3}, std::vector<size_t>{2, 1, 0, 3}, std::vector<size_t>{0, 3, 2, 1}};

const std::vector<std::vector<size_t>> inputShapesMerged4d = {std::vector<size_t>{6, 4, 8, 512},
                                                              std::vector<size_t>{12, 7, 12, 4}};

const auto paramsMerged4d = testing::Combine(
        testing::ValuesIn(orderMerged4d), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesMerged4d), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TransposeMerged4d, TransposeLayerTest_NPU3720, paramsMerged4d,
                        TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test permutation decomposition  ============= */

const std::vector<std::vector<size_t>> complex5DReorder = {std::vector<size_t>{0, 2, 4, 1, 3}};
const std::vector<std::vector<size_t>> inputShapeBatched5D = {std::vector<size_t>{3, 128, 4, 128, 4}};

const auto paramsPermuteDecomposition = testing::Combine(
        testing::ValuesIn(complex5DReorder), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapeBatched5D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_Transpose_Permutation_Decomposition, TransposeLayerTest_NPU3720,
                        paramsPermuteDecomposition, TransposeLayerTest_NPU3720::getTestCaseName);

}  // namespace
