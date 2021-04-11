// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSplitSoftmaxLayerTest : public SplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        int64_t axis;
        size_t numSplits;
        std::vector<size_t> inputShape, outIndices;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outIndices,
                 targetDevice) = this->GetParam();
        if (outIndices.empty()) {
            for (size_t i = 0; i < numSplits; ++i) {
                outIndices.push_back(i);
            }
        }
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto split = std::dynamic_pointer_cast<ngraph::opset5::Split>(
                ngraph::builder::makeSplit(paramOuts[0], ngPrc, numSplits, axis));

        if(axis < 0) {
            axis += inputShape.size();
        }
        ngraph::ResultVector results;
        results.reserve(outIndices.size());
        for (const auto i : outIndices) {
            const auto softMax = std::make_shared<ngraph::opset1::Softmax>(split->output(i), axis);
            results.emplace_back(std::make_shared<ngraph::opset1::Result>(softMax));
        }
        function = std::make_shared<ngraph::Function>(results, params, "split");
    }
};

TEST_P(KmbSplitSoftmaxLayerTest, SubgraphCompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbSplitSoftmaxLayerTest, DISABLED_SubgraphCompareWithRefs_MCM) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(Split, KmbSplitSoftmaxLayerTest,
                        ::testing::Combine(::testing::Values(2, 3), ::testing::Values(-2, 3),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                                           ::testing::Values(InferenceEngine::SizeVector({1, 6, 12, 24})),
                                           ::testing::Values(InferenceEngine::SizeVector({})),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        SplitLayerTest::getTestCaseName);
}  // namespace
