//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/depth_to_space.hpp"

namespace LayerTestsDefinitions {

class KmbDepthToSpaceLayerTest : public DepthToSpaceLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbDepthToSpaceLayerTest, CompareWithRefs) {
    Run();
}
TEST_P(KmbDepthToSpaceLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class DepthToSpaceLayerTest_MLIR_VPU3720 :
        public DepthToSpaceLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(DepthToSpaceLayerTest_MLIR_VPU3720, CompareWithRefs_MLIR_VPU3720) {
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
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP16,  // CPU-plugin has parameter I16, but KMB does not
};                                         // support it. So I16 is changed to FP16.

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapesBS2 = {
        {1, 4, 1, 1}, {1, 4, 2, 2}, {1, 4, 3, 3}, {2, 32, 3, 3}, {2, 16, 5, 4}};
// {1, 8, 1, 1, 1}, {1, 8, 2, 2, 2}, {1, 8, 3, 3, 3}, {2, 32, 3, 3, 3}, {2, 16, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but VPUx-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: vpux-plugin/src/utils/dims_parser.cpp

const auto DepthToSpaceBS2 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS2), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_DepthToSpaceBS2, KmbDepthToSpaceLayerTest, DepthToSpaceBS2,
                         KmbDepthToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapesBS3 = {
        {1, 9, 1, 1}, {1, 9, 2, 2}, {1, 9, 3, 3}, {2, 36, 3, 3}, {2, 27, 5, 4}};
// {1, 27, 1, 1, 1}, {1, 27, 2, 2, 2}, {1, 27, 3, 3, 3}, {2, 108, 3, 3, 3}, {2, 54, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but vpux-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: vpux-plugin/src/utils/dims_parser.cpp

const auto DepthToSpaceBS3 = ::testing::Combine(
        ::testing::ValuesIn(inputShapesBS3), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes),
        ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_DepthToSpaceBS3, KmbDepthToSpaceLayerTest, DepthToSpaceBS3,
                         KmbDepthToSpaceLayerTest::getTestCaseName);

const auto DepthToSpaceBS2_PRECOMMIT_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 4, 3, 3}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(2), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS2_VPU3720, DepthToSpaceLayerTest_MLIR_VPU3720,
                         DepthToSpaceBS2_PRECOMMIT_VPU3720, DepthToSpaceLayerTest_MLIR_VPU3720::getTestCaseName);

const auto DepthToSpaceBS3_PRECOMMIT_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 9, 3, 3}), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(3), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS3_VPU3720, DepthToSpaceLayerTest_MLIR_VPU3720,
                         DepthToSpaceBS3_PRECOMMIT_VPU3720, DepthToSpaceLayerTest_MLIR_VPU3720::getTestCaseName);

const auto smoke_DepthToSpaceBS4_with_tiling_VPU3720 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 48, 80, 80}), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::ValuesIn(modes), ::testing::Values(4), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_tiling_VPU3720, DepthToSpaceLayerTest_MLIR_VPU3720,
                         smoke_DepthToSpaceBS4_with_tiling_VPU3720,
                         DepthToSpaceLayerTest_MLIR_VPU3720::getTestCaseName);
}  // namespace
