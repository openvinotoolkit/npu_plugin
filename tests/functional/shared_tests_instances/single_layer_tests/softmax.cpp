//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/softmax.hpp"

#include <vector>

#include "vpux_layer_test.hpp"

namespace ov {
namespace test {
namespace subgraph {

using namespace VPUXLayerTestsUtils;

class VPUXSoftMaxLayerTest : public SoftMaxLayerTest, virtual public VPUXLayerTestsCommon {
    SkipMessage SkipBeforeLoad() override {
        return vpux::None;
    }
};
class VPUXSoftMaxLayerTest_VPU3720 : public SoftMaxLayerTest, virtual public VPUXLayerTestsCommon {};

class VPUXSoftMaxTilingTest_VPU3720 : public VPUXSoftMaxLayerTest {};

TEST_P(VPUXSoftMaxLayerTest, MLIR) {
    abs_threshold = 1e-3;
    useCompilerMLIR();
    run();
}

TEST_P(VPUXSoftMaxLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    abs_threshold = 1e-3;
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    run();
}

TEST_P(VPUXSoftMaxTilingTest_VPU3720, MLIR_VPU3720) {
    abs_threshold = 1e-3;
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    run();
}

const std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
};

const std::vector<InferenceEngine::Layout> inLayouts2D = {
        InferenceEngine::Layout::NC,
};

const std::vector<ov::Shape> inShapes2D = {
        {1, 100},
        {100, 1},
        {10, 10},
};

const std::vector<size_t> axis2D = {0, 1};

const std::vector<ElementType> inputPrecisions = {
        ov::element::f16,
};

const std::vector<ElementType> outputPrecisions = {
        ov::element::f16,
};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)), testing::ValuesIn(axis2D),
        testing::Values(testPlatformTargetDevice), testing::Values(Config{}));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_SoftMax2D, VPUXSoftMaxLayerTest, params2D,
                        SoftMaxLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_SoftMax2D, VPUXSoftMaxLayerTest_VPU3720, params2D,
                        SoftMaxLayerTest::getTestCaseName);

const std::vector<ov::Shape> inShapes4D = {
        {1, 2, 204, 62}, {1, 12, 2, 1444}, {1, 4, 1, 1}, {1, 1000, 1, 1}, {300, 21, 1, 1}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)), testing::ValuesIn(axis4D),
        testing::Values(testPlatformTargetDevice), testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_SoftMax4D, VPUXSoftMaxLayerTest, params4D,
                        SoftMaxLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_SoftMax4D, VPUXSoftMaxLayerTest_VPU3720, params4D,
                        SoftMaxLayerTest::getTestCaseName);

const auto precommit_params4D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})), testing::ValuesIn(axis4D),
        testing::Values(testPlatformTargetDevice), testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_SoftMax4D, VPUXSoftMaxLayerTest, precommit_params4D,
                        SoftMaxLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_SoftMax4D, VPUXSoftMaxLayerTest_VPU3720, precommit_params4D,
                        SoftMaxLayerTest::getTestCaseName);

const std::vector<ov::Shape> inShapes2DCasses = {{32, 76}, {16800, 2}};

const std::vector<size_t> axis2DCasses = {1};

const auto params2DCasses = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2DCasses)),
        testing::ValuesIn(axis2DCasses), testing::Values(testPlatformTargetDevice), testing::Values(Config{}));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_SoftMax2DCasses, VPUXSoftMaxLayerTest_VPU3720, params2DCasses,
                        SoftMaxLayerTest::getTestCaseName);

// ------ Test tiling functionality ------

const std::vector<ov::Shape> inShapes = {{1, 20, 64, 512}};
const std::vector<size_t> axis = {1};

const auto paramsTilingCasesVPU3720 = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), testing::ValuesIn(axis),
        testing::Values(testPlatformTargetDevice), testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TilingSoftMax, VPUXSoftMaxTilingTest_VPU3720, paramsTilingCasesVPU3720,
                        SoftMaxLayerTest::getTestCaseName);

}  // namespace subgraph
}  // namespace test
}  // namespace ov
