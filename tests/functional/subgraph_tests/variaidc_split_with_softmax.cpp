// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/variadic_split.hpp"
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbVariadicSplitLayerTest : public VariadicSplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        std::size_t axis;
        std::vector<size_t> inputShape, numSplits;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axis, netPrecision, KmbLayerTestsCommon::inPrc, KmbLayerTestsCommon::outPrc, KmbLayerTestsCommon::inLayout, KmbLayerTestsCommon::outLayout, inputShape,
                 KmbLayerTestsCommon::targetDevice) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        
        auto variadicSplit = std::dynamic_pointer_cast<ngraph::opset3::VariadicSplit>(ngraph::builder::makeVariadicSplit(paramOuts[0], numSplits,
                axis));

        ngraph::ResultVector results;
        results.reserve(numSplits.size());
        for (int i = 0; i < numSplits.size(); i++) {
            const auto softMax = std::make_shared<ngraph::opset1::Softmax>(variadicSplit->output(i), axis);
            results.emplace_back(std::make_shared<ngraph::opset1::Result>(softMax));
        }

        KmbLayerTestsCommon::function = std::make_shared<ngraph::Function>(results, params, "variadicSplit");
    }
};

TEST_P(KmbVariadicSplitLayerTest, SubgraphCompareWithRefs_MLIR) {
    useCompilerMLIR();
    KmbLayerTestsCommon::Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::SizeVector> inputShapes = {InferenceEngine::SizeVector{1, 144, 30, 40}};

const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP32;

const std::vector<size_t> numSplits = {64, 48, 32};

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplit, KmbVariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(numSplits), ::testing::Values(1),
                                           ::testing::Values(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbVariadicSplitLayerTest::getTestCaseName);

}  // namespace
