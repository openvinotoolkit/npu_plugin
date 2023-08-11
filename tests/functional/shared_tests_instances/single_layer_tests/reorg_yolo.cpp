//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reorg_yolo.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXReorgYoloLayerTest : public ReorgYoloLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXReorgYoloLayerTest_VPU3700 : public VPUXReorgYoloLayerTest {
    void SkipBeforeLoad() override {
        ngraph::Shape inputShape;
        std::tie(inputShape, std::ignore, std::ignore, std::ignore) = GetParam();
        auto inN = inputShape[0];
        if (inN != 1) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports N=1 shape, got " + std::to_string(inN));
        }
    }
};

class VPUXReorgYoloLayerTest_VPU3720 : public VPUXReorgYoloLayerTest {};

TEST_P(VPUXReorgYoloLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXReorgYoloLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ngraph::Shape> inputShapesA = {
        ngraph::Shape{1, 64, 26, 26},  // openvino eg
        ngraph::Shape{1, 4, 4, 4}, ngraph::Shape{1, 8, 4, 4}, ngraph::Shape{2, 8, 4, 4},

        // fails:
        // [Track number: E#29273]
        // ngraph::Shape{1, 62, 14, 14},
        // ngraph::Shape{1, 62, 34, 24},
        // ngraph::Shape{1, 24, 34, 62},
        // ngraph::Shape{1, 26, 64, 26},
};

const std::vector<size_t> stridesA = {2};

const std::vector<ngraph::Shape> inputShapesB = {
        ngraph::Shape{1, 9, 3, 3},
};

const std::vector<size_t> stridesB = {3};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(smoke_ReorgYolo_a, VPUXReorgYoloLayerTest_VPU3700,
                        testing::Combine(testing::ValuesIn(inputShapesA), testing::ValuesIn(stridesA),
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXReorgYoloLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReorgYolo_b, VPUXReorgYoloLayerTest_VPU3700,
                        testing::Combine(testing::ValuesIn(inputShapesB), testing::ValuesIn(stridesB),
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXReorgYoloLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(precommit_smoke_ReorgYolo_a, VPUXReorgYoloLayerTest_VPU3720,
                        testing::Combine(testing::ValuesIn(inputShapesA), testing::ValuesIn(stridesA),
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXReorgYoloLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReorgYolo_b, VPUXReorgYoloLayerTest_VPU3720,
                        testing::Combine(testing::ValuesIn(inputShapesB), testing::ValuesIn(stridesB),
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXReorgYoloLayerTest_VPU3720::getTestCaseName);

}  // namespace
