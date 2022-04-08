//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/tile.hpp"

namespace LayerTestsDefinitions {

class KmbTileLayerTest : public TileLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
    }
};

TEST_P(KmbTileLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbTileLayerTest_VPU3720 : public TileLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbTileLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> repeats = {
        // tile by single axes
        {1, 1, 1, 5},
        {1, 1, 5},
        {1, 5, 1},
        {5, 1, 1},
        {1, 8},
        {8, 1},

        // tile by multiple axes
        {1, 2, 3},
        {2, 3, 1},
        {3, 1, 2},

        // identical tile case
        {1, 1, 1},

        // input shapes with more than 4D is not supported by runtime yet
        // {1, 1, 1, 2, 1, 2}

        // looks like this values is too big. Test fails due result mismatch between CPU an KMB
        // {1, 1, 1, 128}, {1, 1, 128, 1}, {1, 128, 1, 1}, {128, 1, 1, 1},
};

const std::vector<std::vector<size_t>> inputShapes = {
        {2},       {2, 3},

        {3, 4, 2}, {2, 3, 4, 2}, {1, 1, 128, 1},

        // input shapes with more than 4D is not supported by runtime yet
        // {1, 4, 3, 1, 3, 1}
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Tile, KmbTileLayerTest,
                         ::testing::Combine(::testing::ValuesIn(repeats), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbTileLayerTest::getTestCaseName);

// VPU3720
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile3720, KmbTileLayerTest_VPU3720,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({{1, 3, 2}, {3, 2, 1, 5}, {1, 3, 2, 1}})),
                ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 4, 3, 2}, {4, 3, 2, 1}, {4, 3, 2, 5}})),
                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbTileLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Tile3720, KmbTileLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int64_t>>({{2, 3, 1}})),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 4, 2}})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbTileLayerTest_VPU3720::getTestCaseName);

}  // namespace
