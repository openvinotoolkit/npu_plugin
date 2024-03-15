// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "shared_test_classes/single_layer/variadic_split.hpp"
#include <memory>
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "ov_models/builders.hpp"
#include "vpu_ov1_layer_test.hpp"

std::shared_ptr<ngraph::Node> makeVariadicSplit(const ngraph::Output<ov::Node>& in, const std::vector<size_t> numSplits,
                                                int32_t axis) {
    auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32, ngraph::Shape{},
                                                                  std::vector<int32_t>{axis});
    auto numSplit = std::make_shared<ngraph::opset3::Constant>(ngraph::element::u64, ngraph::Shape{numSplits.size()},
                                                               numSplits);
    return std::make_shared<ngraph::opset3::VariadicSplit>(in, splitAxisOp, numSplit);
}

namespace LayerTestsDefinitions {
class VariadicSplitLayerTestCommon :
        public VariadicSplitLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VariadicSplitLayerTest_NPU3700 : public VariadicSplitLayerTestCommon {};
class VariadicSplitLayerTest_NPU3720 : public VariadicSplitLayerTestCommon {};

class VariadicSplitLayerTestAxisInt32_NPU3720 : public VariadicSplitLayerTestCommon {
    void SetUp() override {
        int32_t axisInt32;
        std::vector<size_t> inputShape, numSplits;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axisInt32, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
                this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto variadicSplit = std::dynamic_pointer_cast<ngraph::opset3::VariadicSplit>(
                makeVariadicSplit(params[0], numSplits, axisInt32));
        ngraph::ResultVector results;
        for (int i = 0; i < numSplits.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset3::Result>(variadicSplit->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "VariadicSplit");
    }
};

TEST_P(VariadicSplitLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VariadicSplitLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VariadicSplitLayerTestAxisInt32_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::SizeVector> inputShapes = {InferenceEngine::SizeVector{1, 144, 30, 40}};

const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP32;

const std::vector<size_t> numSplits = {64, 48, 32};

const auto variadicSplitParams0 =
        testing::Combine(::testing::Values(numSplits),                                // numSplits
                         ::testing::Values(1),                                        // axis
                         ::testing::Values(netPrecisions),                            // netPrecision
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::ValuesIn(inputShapes),                            // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams1 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams2 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),   // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams3 =
        testing::Combine(::testing::Values(std::vector<size_t>{2, 4, 4}),                 // numSplits
                         ::testing::Values(0, 1, 2, 3),                                   // axis
                         ::testing::Values(netPrecisions),                                // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),                 // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),                 // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{10, 10, 10, 10}),  // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto variadicSplitParams4 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                // numSplits
                         ::testing::Values(-1),                                       // axis
                         ::testing::Values(netPrecisions),                            // netPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // inPrc
                         ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  // outPrc
                         ::testing::Values(InferenceEngine::Layout::ANY),             // inLayout
                         ::testing::Values(InferenceEngine::Layout::ANY),             // outLayout
                         ::testing::Values(InferenceEngine::SizeVector{1, 4, 2}),     // inputShapes
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// Tracking number [E#85137]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_VariadicSplit, VariadicSplitLayerTest_NPU3700, variadicSplitParams0,
                        VariadicSplitLayerTest_NPU3700::getTestCaseName);

/* ============= Negative Axis ============= */

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis, VariadicSplitLayerTest_NPU3700, variadicSplitParams1,
                        VariadicSplitLayerTest_NPU3700::getTestCaseName);

/* ============= NPU3720  ============= */
INSTANTIATE_TEST_CASE_P(smoke_VariadicSplit, VariadicSplitLayerTest_NPU3720, variadicSplitParams0,
                        VariadicSplitLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis0, VariadicSplitLayerTest_NPU3720, variadicSplitParams1,
                        VariadicSplitLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis1, VariadicSplitLayerTest_NPU3720, variadicSplitParams2,
                        VariadicSplitLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitPosAxis, VariadicSplitLayerTest_NPU3720, variadicSplitParams3,
                        VariadicSplitLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_VariadicSplitNegAxis, VariadicSplitLayerTest_NPU3720, variadicSplitParams4,
                        VariadicSplitLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitInt32Axis, VariadicSplitLayerTestAxisInt32_NPU3720, variadicSplitParams3,
                        VariadicSplitLayerTestAxisInt32_NPU3720::getTestCaseName);

}  // namespace
