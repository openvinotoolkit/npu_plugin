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
        InputShape inShapes;
        size_t axisInd;
        std::tie(std::ignore, inType, outType, inShapes, axisInd, std::ignore, std::ignore) = GetParam();

        if (isCompilerMCM()) {
            // [Track number: S#44702]
            if (inType == ov::element::f32 || outType == ov::element::f32) {
                return {"SoftMax with FP32 input/output hangs on graph loading"};
            }

            // [Track number: S#40296]
            for (const auto& shape : inShapes.second) {
                if (shape.at(axisInd) == 1) {
                    return {"SoftMax over dim==1 fails during blob parsing"};
                }
            }
        }

        return vpux::None;
    }
};

TEST_P(VPUXSoftMaxLayerTest, MCM) {
    abs_threshold = 1e-3;
    run();
}

TEST_P(VPUXSoftMaxLayerTest, MLIR) {
    abs_threshold = 1e-3;
    useCompilerMLIR();
    run();
}

TEST_P(VPUXSoftMaxLayerTest, MLIR_VPUX37XX) {
    abs_threshold = 1e-3;
    useCompilerMLIR();
    setPlatformVPUX37XX();
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

INSTANTIATE_TEST_CASE_P(smoke_SoftMax2D, VPUXSoftMaxLayerTest, params2D, SoftMaxLayerTest::getTestCaseName);

const std::vector<ov::Shape> inShapes4D = {{1, 2, 204, 62}, {1, 12, 2, 1444}, {1, 2, 72, 10},
                                           {1, 4, 1, 1},    {1, 1000, 1, 1},  {300, 21, 1, 1}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)), testing::ValuesIn(axis4D),
        testing::Values(testPlatformTargetDevice), testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_CASE_P(smoke_SoftMax4D, VPUXSoftMaxLayerTest, params4D, SoftMaxLayerTest::getTestCaseName);

}  // namespace subgraph
}  // namespace test
}  // namespace ov
