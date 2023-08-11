//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include "shared_test_classes/single_layer/concat.hpp"

namespace ConcatSoftmaxSubGraphTestsDefinitions {
class VPUXConcatSoftmaxSubGraphTest_VPU3700 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public LayerTestsDefinitions::ConcatLayerTest {
    void SetUp() override {
        int axis;
        std::vector<std::vector<size_t>> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, inputShape);
        auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);
        auto softMax = std::make_shared<ngraph::opset1::Softmax>(concat, axis);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};
        function = std::make_shared<ngraph::Function>(results, params, "concat_softmax");

        threshold = 0.1f;
    }
};

TEST_P(VPUXConcatSoftmaxSubGraphTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

// Note: VPUX-plugin does not support batch-size > 1.

// 4d cases
std::vector<int> axes4d = {1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes4d = {
        {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        {{1, 10, 10, 10}, {1, 10, 10, 10}, {1, 10, 10, 10}},
        {{1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke4d_tensors, VPUXConcatSoftmaxSubGraphTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes4d), ::testing::ValuesIn(inShapes4d),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConcatSoftmaxSubGraphTest_VPU3700::getTestCaseName);

// 3d cases
std::vector<int> axes3d = {1, 2};
std::vector<std::vector<std::vector<size_t>>> inShapes3d = {
        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
        {{1, 10, 33}, {1, 10, 33}, {1, 10, 33}, {1, 10, 33}},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke3d_tensors, VPUXConcatSoftmaxSubGraphTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes3d), ::testing::ValuesIn(inShapes4d),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConcatSoftmaxSubGraphTest_VPU3700::getTestCaseName);

// Check parameters from squeezenet1_1
std::vector<int> axes_squeeznet1_1 = {1};

std::vector<std::vector<std::vector<size_t>>> inShapes_squeeznet1_1 = {{{1, 64, 56, 56}, {1, 64, 56, 56}},
                                                                       {{1, 192, 14, 14}, {1, 192, 14, 14}},
                                                                       {{1, 128, 28, 28}, {1, 128, 28, 28}},
                                                                       {{1, 256, 14, 14}, {1, 256, 14, 14}}};

INSTANTIATE_TEST_SUITE_P(smoke_squeeznet1_1_tensors, VPUXConcatSoftmaxSubGraphTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes_squeeznet1_1),
                                            ::testing::ValuesIn(inShapes_squeeznet1_1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConcatSoftmaxSubGraphTest_VPU3700::getTestCaseName);

}  // namespace ConcatSoftmaxSubGraphTestsDefinitions
