//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/softmax.hpp"
#include <algorithm>
#include <vector>
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class VPUXSoftMaxLayerTest : public SoftMaxLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(VPUXSoftMaxLayerTest, VPU3700_SW) {
    abs_threshold = 1e-3;
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(VPUXSoftMaxLayerTest, VPU3700_HW) {
    abs_threshold = 1e-3;
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(VPUXSoftMaxLayerTest, VPU3720_SW) {
    abs_threshold = 0.01;
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(VPUXSoftMaxLayerTest, VPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace ov::test::subgraph

namespace {

using namespace ov::test::subgraph;

const std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> inputPrecisions = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> outputPrecisions = {
        ov::element::f16,
};

//
// Input 2D
//

const std::vector<ov::Shape> inShapes2D = {
        {1, 100}, {100, 1}, {10, 10}, {32, 76}, {72, 2},
};

const std::vector<size_t> axis2D = {0, 1};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)), testing::ValuesIn(axis2D),
        testing::Values(targetDevice), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_SoftMax2D, VPUXSoftMaxLayerTest, params2D, SoftMaxLayerTest::getTestCaseName);

//
// Input 3D
//

const std::vector<ov::Shape> inShapes3D = {{1, 4300, 2}, {8, 182, 182}};

const std::vector<size_t> axis3D = {2};

const auto params3D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes3D)), testing::ValuesIn(axis3D),
        testing::Values(targetDevice), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_SoftMax3D, VPUXSoftMaxLayerTest, params3D, SoftMaxLayerTest::getTestCaseName);

//
// Input 4D
//

const std::vector<ov::Shape> inShapes4D = {{1, 2, 108, 60}, {1, 12, 2, 148}, {1, 4, 1, 1},
                                           {1, 100, 1, 1},  {300, 21, 1, 1}, {1, 2, 48, 2}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)), testing::ValuesIn(axis4D),
        testing::Values(targetDevice), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_SoftMax4D, VPUXSoftMaxLayerTest, params4D, SoftMaxLayerTest::getTestCaseName);

const auto precommit_params4D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})), testing::ValuesIn(axis4D),
        testing::Values(targetDevice), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_precommit_SoftMax4D, VPUXSoftMaxLayerTest, precommit_params4D,
                        SoftMaxLayerTest::getTestCaseName);

//
// Test tiling functionality
//

const std::vector<ov::Shape> inShapes = {{1, 20, 64, 512}};
const std::vector<size_t> axis = {1};

const auto paramsTilingCasesVPU3720 = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), testing::ValuesIn(axis),
        testing::Values(targetDevice), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_CASE_P(smoke_TilingSoftMax, VPUXSoftMaxLayerTest, paramsTilingCasesVPU3720,
                        SoftMaxLayerTest::getTestCaseName);

}  // namespace
