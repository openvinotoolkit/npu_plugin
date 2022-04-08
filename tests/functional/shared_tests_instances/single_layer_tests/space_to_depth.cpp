//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/space_to_depth.hpp"

namespace LayerTestsDefinitions {

class KmbSpaceToDepthLayerTest : public SpaceToDepthLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbSpaceToDepthLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbSpaceToDepthLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class SpaceToDepthLayerTest_MLIR_VPU3720 :
        public SpaceToDepthLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(SpaceToDepthLayerTest_MLIR_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,  // value from CPU-plugin I16 is changed for FP16
        InferenceEngine::Precision::U8};

const std::vector<SpaceToDepth::SpaceToDepthMode> modes = {SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                           SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapesBS2 = {
        {1, 1, 2, 2}, {1, 1, 4, 4}, {1, 1, 6, 6}, {2, 8, 6, 6}, {2, 4, 10, 8}};

const auto SpaceToDepthBS2 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS2), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_SpaceToDepthBS2, KmbSpaceToDepthLayerTest, SpaceToDepthBS2,
                         KmbSpaceToDepthLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapesBS3 = {
        {1, 1, 3, 3}, {1, 1, 6, 6}, {1, 1, 9, 9}, {2, 4, 9, 9}, {2, 3, 15, 12}};

const auto SpaceToDepthBS3 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS3), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_SpaceToDepthBS3, KmbSpaceToDepthLayerTest, SpaceToDepthBS3,
                         KmbSpaceToDepthLayerTest::getTestCaseName);

const auto SpaceToDepthBS2_PRECOMMIT_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 3 * 4, 3 * 4}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS2_VPU3720, SpaceToDepthLayerTest_MLIR_VPU3720,
                         SpaceToDepthBS2_PRECOMMIT_VPU3720, SpaceToDepthLayerTest_MLIR_VPU3720::getTestCaseName);

const auto SpaceToDepthBS3_PRECOMMIT_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 3 * 3, 3 * 3}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS3_VPU3720, SpaceToDepthLayerTest_MLIR_VPU3720,
                         SpaceToDepthBS3_PRECOMMIT_VPU3720, SpaceToDepthLayerTest_MLIR_VPU3720::getTestCaseName);

const auto smoke_SpaceToDepthBS4_with_tiling_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 48, 160, 80}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(4), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_SpaceToDepth_with_tiling_VPU3720, SpaceToDepthLayerTest_MLIR_VPU3720,
                         smoke_SpaceToDepthBS4_with_tiling_VPU3720,
                         SpaceToDepthLayerTest_MLIR_VPU3720::getTestCaseName);

}  // namespace
