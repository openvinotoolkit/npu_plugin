//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/tile.hpp"

namespace LayerTestsDefinitions {

class VPUXTileLayerTest : public TileLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXTileLayerTest_VPU3700 : public VPUXTileLayerTest {
    void SkipBeforeLoad() override {
    }
};

class VPUXTileLayerTest_VPU3720_SW : public VPUXTileLayerTest {};
class VPUXTileLayerTest_VPU3720_HW : public VPUXTileLayerTest {};

TEST_P(VPUXTileLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXTileLayerTest_VPU3720_SW, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXTileLayerTest_VPU3720_HW, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::U8};

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

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Tile, VPUXTileLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(repeats), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXTileLayerTest_VPU3700::getTestCaseName);

const auto tileParamsVPUX = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>({{1, 3, 2}, {3, 2, 1, 5}, {1, 3, 2, 1}})),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 4, 3, 2}, {4, 3, 2, 1}, {4, 3, 2, 5}})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto tileParamsPrecommitVPUX = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>({{2, 3, 1}})), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 4, 2}})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

// VPU3720

INSTANTIATE_TEST_SUITE_P(smoke_Tile3720, VPUXTileLayerTest_VPU3720_SW, tileParamsVPUX,
                         VPUXTileLayerTest_VPU3720_SW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Tile3720, VPUXTileLayerTest_VPU3720_SW, tileParamsPrecommitVPUX,
                         VPUXTileLayerTest_VPU3720_SW::getTestCaseName);

// VPU3720 - tiling

// [Tracking number E#78540]
// case 1: tile on two dimensions
INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Tile3720_tiling_1, VPUXTileLayerTest_VPU3720_HW,
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int64_t>>({{1, 2, 3, 3}})),  // repeats_values
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1, 2880, 50}})),  // input_shape
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXTileLayerTest_VPU3720_HW::getTestCaseName);

// [Tracking number E#78540]
// case 2: repeats values aren't 1
INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Tile3720_tiling_2, VPUXTileLayerTest_VPU3720_HW,
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int64_t>>({{2, 2, 3, 3}})),  // repeats_values
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>({{3, 2, 723, 25}})),  // input_shape
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXTileLayerTest_VPU3720_HW::getTestCaseName);

// case 3: repeats values may be 1
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile3720_tiling_3, VPUXTileLayerTest_VPU3720_HW,
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<int64_t>>({{3, 1, 3, 2}})),  // repeats_values
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 3, 360, 50}})),  // input_shape
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXTileLayerTest_VPU3720_HW::getTestCaseName);

// model case: tensor<1x32x1x1xf16> -> tensor<1x32x1x65536xf16> , NHWC

INSTANTIATE_TEST_SUITE_P(
        smoke_Tile3720_tiling_4, VPUXTileLayerTest_VPU3720_HW,
        ::testing::Combine(
                ::testing::ValuesIn(std::vector<std::vector<int64_t>>({{1, 1, 1, 65536}})),  // repeats_values
                ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::NHWC),
                ::testing::Values(InferenceEngine::Layout::NHWC),
                ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 32, 1, 1}})),  // input_shape
                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXTileLayerTest_VPU3720_HW::getTestCaseName);

}  // namespace
