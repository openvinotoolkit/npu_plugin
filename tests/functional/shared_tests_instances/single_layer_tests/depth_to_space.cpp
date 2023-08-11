//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/depth_to_space.hpp"

namespace LayerTestsDefinitions {

class VPUXDepthToSpaceLayerTest : public DepthToSpaceLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXDepthToSpaceLayerTest_VPU3700 : public VPUXDepthToSpaceLayerTest {};
class VPUXDepthToSpaceLayerTest_VPU3720 : public VPUXDepthToSpaceLayerTest {};

TEST_P(VPUXDepthToSpaceLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXDepthToSpaceLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP16,  // CPU-plugin has parameter I16, but KMB does not
};                                         // support it. So I16 is changed to FP16.

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapesBS2 = {
        {1, 4, 1, 1}, {1, 4, 2, 2}, {1, 4, 3, 3}, {2, 32, 3, 3}, {2, 16, 5, 4}};
// {1, 8, 1, 1, 1}, {1, 8, 2, 2, 2}, {1, 8, 3, 3, 3}, {2, 32, 3, 3, 3}, {2, 16, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but VPUX-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: VPUX-plugin/src/utils/dims_parser.cpp

const auto DepthToSpaceBS2 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS2), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_DepthToSpaceBS2, VPUXDepthToSpaceLayerTest_VPU3700, DepthToSpaceBS2,
                         VPUXDepthToSpaceLayerTest_VPU3700::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapesBS3 = {
        {1, 9, 1, 1}, {1, 9, 2, 2}, {1, 9, 3, 3}, {2, 36, 3, 3}, {2, 27, 5, 4}};
// {1, 27, 1, 1, 1}, {1, 27, 2, 2, 2}, {1, 27, 3, 3, 3}, {2, 108, 3, 3, 3}, {2, 54, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but VPUX-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: VPUX-plugin/src/utils/dims_parser.cpp

const auto DepthToSpaceBS3 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS3), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_DepthToSpaceBS3, VPUXDepthToSpaceLayerTest_VPU3700, DepthToSpaceBS3,
                         VPUXDepthToSpaceLayerTest_VPU3700::getTestCaseName);

const auto DepthToSpaceBS2_PRECOMMIT = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 4, 3, 3}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto DepthToSpaceBS3_PRECOMMIT = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 9, 3, 3}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto smoke_DepthToSpaceBS4_with_tiling = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 48, 80, 80}), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::ValuesIn(modes), ::testing::Values(4), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

/* ============= VPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS2_VPU3720, VPUXDepthToSpaceLayerTest_VPU3720,
                         DepthToSpaceBS2_PRECOMMIT, VPUXDepthToSpaceLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS3_VPU3720, VPUXDepthToSpaceLayerTest_VPU3720,
                         DepthToSpaceBS3_PRECOMMIT, VPUXDepthToSpaceLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_tiling_VPU3720, VPUXDepthToSpaceLayerTest_VPU3720,
                         smoke_DepthToSpaceBS4_with_tiling, VPUXDepthToSpaceLayerTest_VPU3720::getTestCaseName);

}  // namespace
