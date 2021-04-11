// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include "shared_test_classes/single_layer/concat.hpp"
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>


namespace ConcatSoftmaxSubGraphTestsDefinitions {
    class KmbConcatSoftmaxSubGraphTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                         public LayerTestsDefinitions::ConcatLayerTest {
        void SetUp() override {
            int axis;
            std::vector<std::vector<size_t>> inputShape;
            InferenceEngine::Precision netPrecision;
            std::tie(axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
            auto params = ngraph::builder::makeParams(ngPrc, inputShape);
            auto paramOuts = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
            auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);
            auto softMax = std::make_shared<ngraph::opset1::Softmax>(concat, axis);
            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};
            function = std::make_shared<ngraph::Function>(results, params, "concat_softmax");

            threshold = 0.1f;
        }
    };

    TEST_P(KmbConcatSoftmaxSubGraphTest, CompareWithRefs_MLIR) {
        useCompilerMLIR();
        Run();
    }

    std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
    };

// Note: KMB-plugin does not support batch-size > 1.

// 4d cases
    InferenceEngine::SizeVector axes4d = {1, 2, 3};
    std::vector<std::vector<std::vector<size_t>>> inShapes4d = {
            {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
            {{1, 10, 10, 10}, {1, 10, 10, 10}, {1, 10, 10, 10}},
            {{1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}},
    };

    INSTANTIATE_TEST_CASE_P(smoke4d_tensors, KmbConcatSoftmaxSubGraphTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(axes4d),
                                    ::testing::ValuesIn(inShapes4d),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConcatSoftmaxSubGraphTest::getTestCaseName);


// 3d cases
    InferenceEngine::SizeVector axes3d = {1, 2};
    std::vector<std::vector<std::vector<size_t>>> inShapes3d = {
            {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
            {{1, 10, 33}, {1, 10, 33}, {1, 10, 33}, {1, 10, 33}},
    };

    INSTANTIATE_TEST_CASE_P(smoke3d_tensors, KmbConcatSoftmaxSubGraphTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(axes3d),
                                    ::testing::ValuesIn(inShapes4d),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConcatSoftmaxSubGraphTest::getTestCaseName);

// Check parameters from squeezenet1_1
    InferenceEngine::SizeVector axes_squeeznet1_1 = {1};

    std::vector<std::vector<std::vector<size_t>>> inShapes_squeeznet1_1 = {
            {{1, 64, 56, 56}, {1, 64, 56, 56}},
            {{1, 192, 14, 14}, {1, 192, 14, 14}},
            {{1, 128, 28, 28}, {1, 128, 28, 28}},
            {{1, 256, 14, 14}, {1, 256, 14, 14}}
    };

    INSTANTIATE_TEST_CASE_P(squeeznet1_1_tensors, KmbConcatSoftmaxSubGraphTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(axes_squeeznet1_1),
                                    ::testing::ValuesIn(inShapes_squeeznet1_1),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(InferenceEngine::Layout::ANY),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbConcatSoftmaxSubGraphTest::getTestCaseName);

}  // namespace ConcatSoftmaxSubGraphTestsDefinitions
