//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/transpose.hpp"

namespace LayerTestsDefinitions {

class KmbTransposeLayerTest : public TransposeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class KmbTransposeLayerTest_MLIR : public KmbTransposeLayerTest {};
class KmbTransposeLayerTest_VPU3720 : public KmbTransposeLayerTest {};

TEST_P(KmbTransposeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbTransposeLayerTest_MLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbTransposeLayerTest_VPU3720, MLIR_VPU3720) {
    useCompilerMLIR();
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

const auto params2D =
        testing::Combine(testing::ValuesIn(inputOrder2D), testing::ValuesIn(netPrecisions),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inputShapes2D), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

// [Track number: W#7312]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Transpose2D, KmbTransposeLayerTest_MLIR, params2D,
                         KmbTransposeLayerTest::getTestCaseName);

// MLIR 4D instantiation

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 3, 2, 1},

        // [Track number: W#7311]
        // std::vector<size_t>{},
};

const auto params4D =
        testing::Combine(testing::ValuesIn(inputOrder4D), testing::ValuesIn(netPrecisions),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inputShapes4D), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_Transpose4D, KmbTransposeLayerTest_MLIR, params4D,
                         KmbTransposeLayerTest::getTestCaseName);

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
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNCHW, KmbTransposeLayerTest_MLIR, paramsMemPermNCHWtoNHWC,
                        KmbTransposeLayerTest::getTestCaseName);

const auto paramsMemPermInNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::NHWC),
        testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inputShapesMemPerm),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_TransposeMemPermNHWC, KmbTransposeLayerTest_MLIR, paramsMemPermInNHWC,
                        KmbTransposeLayerTest::getTestCaseName);

// ------ VPU3720 ------

const std::vector<std::vector<size_t>> inputOrderVPU3720 = {
        std::vector<size_t>{0, 2, 3, 1},
};

const auto paramsVPU3720 = testing::Combine(
        testing::ValuesIn(inputOrderVPU3720), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesMemPerm), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_precommit_TransposeVPU3720, KmbTransposeLayerTest_VPU3720, paramsVPU3720,
                        KmbTransposeLayerTest::getTestCaseName);

// -------- ND ---------

const std::vector<std::vector<size_t>> shape_5D = {std::vector<size_t>{1, 10, 10, 4, 6},
                                                   std::vector<size_t>{1, 10, 4, 6, 1}};
const std::vector<std::vector<size_t>> reorder_5D = {std::vector<size_t>{4, 1, 2, 3, 0},
                                                     std::vector<size_t>{4, 0, 2, 3, 1}};

const auto paramsVPU3720_5D = testing::Combine(
        testing::ValuesIn(reorder_5D), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(shape_5D), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_TransposeVPU3720_5D, KmbTransposeLayerTest_VPU3720, paramsVPU3720_5D,
                        KmbTransposeLayerTest::getTestCaseName);

}  // namespace
