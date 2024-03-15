//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_depth.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class SpaceToDepthLayerTestCommon :
        public SpaceToDepthLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class SpaceToDepthLayerTest_NPU3700 : public SpaceToDepthLayerTestCommon {};
class SpaceToDepthLayerTest_NPU3720 : public SpaceToDepthLayerTestCommon {};

TEST_P(SpaceToDepthLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(SpaceToDepthLayerTest_NPU3720, HW) {
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

/* ============= NPU 3700 ============= */

const std::vector<std::vector<size_t>> inputShapesBS2 = {
        {1, 1, 2, 2}, {1, 1, 4, 4}, {1, 1, 6, 6}, {2, 8, 6, 6}, {2, 4, 10, 8}};

const std::vector<std::vector<size_t>> inputShapesBS3 = {
        {1, 1, 3, 3}, {1, 1, 6, 6}, {1, 1, 9, 9}, {2, 4, 9, 9}, {2, 3, 15, 12}};

const auto SpaceToDepthBS2 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS2), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto SpaceToDepthBS3 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS3), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS2, SpaceToDepthLayerTest_NPU3700, SpaceToDepthBS2,
                         SpaceToDepthLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS3, SpaceToDepthLayerTest_NPU3700, SpaceToDepthBS3,
                         SpaceToDepthLayerTest_NPU3700::getTestCaseName);

/* ============= NPU 3720 ============= */

const auto SpaceToDepthBS2_PRECOMMIT =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 2, 3 * 4, 3 * 4}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto SpaceToDepthBS3_PRECOMMIT =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 2, 3 * 3, 3 * 3}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto smoke_SpaceToDepthBS4_with_tiling =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 48, 160, 80}), ::testing::ValuesIn(inputPrecisions),
                           ::testing::ValuesIn(modes), ::testing::Values(4),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

/* ============= NPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS2, SpaceToDepthLayerTest_NPU3720, SpaceToDepthBS2_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS3, SpaceToDepthLayerTest_NPU3720, SpaceToDepthBS3_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_SpaceToDepth_with_tiling, SpaceToDepthLayerTest_NPU3720,
                         smoke_SpaceToDepthBS4_with_tiling, SpaceToDepthLayerTest_NPU3720::getTestCaseName);

}  // namespace
