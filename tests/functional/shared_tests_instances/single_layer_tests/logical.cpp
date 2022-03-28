//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/logical.hpp"

#include "kmb_layer_test.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class KmbLogicalLayerTest:
        public LogicalLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        SetupParams();

        ngraph::NodeVector convertedInputs;
        auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        std::shared_ptr<ngraph::Node> logicalNode;
        if (logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT) {
            auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first, inputShapes.second});
            for (const auto& input : inputs) {
                convertedInputs.push_back(std::make_shared<ngraph::opset5::Convert>(input, ngraph::element::boolean));
            }
            logicalNode = ngraph::builder::makeLogical(convertedInputs[0], convertedInputs[1], logicalOpType);
            function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");

        } else {
            auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first});
            ngraph::NodeVector convertedInputs;
            for (const auto& input : inputs) {
                convertedInputs.push_back(std::make_shared<ngraph::opset5::Convert>(input, ngraph::element::boolean));
            }
            logicalNode =
                    ngraph::builder::makeLogical(convertedInputs[0], ngraph::Output<ngraph::Node>(), logicalOpType);
            function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
        }
    }
};

class KmbLogicalLayerTest_MLIR : public KmbLogicalLayerTest {};

TEST_P(KmbLogicalLayerTest_MLIR, CompareWithRefs_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbLogicalLayerTest_MLIR, CompareWithRefs_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t >>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},

        // The same as for eltwise:
        // There are errors at validation step on KMB-board for some input shapes:
        // [Track number: S#51346]
        // {{2, 200}, {{2, 200} }},

        // The same as for eltwise:
        // [Track number: E#15146]
        // Initialization disabled partly
        // {{2, 17, 3, 4}, {{4}, {1, 3, 4}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapesNot = {
        {{5}, {}},
        {{2, 200}, {}},
        {{1, 3, 20}, {}},
        {{1, 17, 3, 4}, {}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::FP16,
};

std::vector<ngraph::helpers::LogicalTypes> logicalOpTypes = {
        ngraph::helpers::LogicalTypes::LOGICAL_AND,
        ngraph::helpers::LogicalTypes::LOGICAL_OR,
        ngraph::helpers::LogicalTypes::LOGICAL_XOR,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {};

const auto LogicalTestParams = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapes)),
        ::testing::ValuesIn(logicalOpTypes),
        ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputsPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

const auto LogicalTestParamsNot = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapesNot)),
        ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_NOT),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputsPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs, KmbLogicalLayerTest_MLIR, LogicalTestParams, LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNot, KmbLogicalLayerTest_MLIR, LogicalTestParamsNot, LogicalLayerTest::getTestCaseName);
}  // namespace
