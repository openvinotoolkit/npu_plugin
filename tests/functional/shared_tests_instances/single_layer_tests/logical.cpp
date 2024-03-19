//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/logical.hpp"

#include "vpu_ov1_layer_test.hpp"

#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

class LogicalLayerTestCommon : public LogicalLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        SetupParams();

        ngraph::NodeVector convertedInputs;
        auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        std::shared_ptr<ngraph::Node> logicalNode;
        if (logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT) {
            ov::ParameterVector inputs{
                    std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first)),
                    std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.second))};
            for (const auto& input : inputs) {
                convertedInputs.push_back(std::make_shared<ov::op::v0::Convert>(input, ngraph::element::boolean));
            }
            logicalNode = ngraph::builder::makeLogical(convertedInputs[0], convertedInputs[1], logicalOpType);
            function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
        } else {
            ov::ParameterVector inputs{
                    std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first))};
            logicalNode = ngraph::builder::makeLogical(inputs[0], ngraph::Output<ngraph::Node>(), logicalOpType);
            function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
        }
    }
};

class LogicalLayerTest_NPU3700 : public LogicalLayerTestCommon {};
class LogicalLayerTest_SW_NPU3720 : public LogicalLayerTestCommon {};
class LogicalLayerTest_HW_NPU3720 : public LogicalLayerTestCommon {};

TEST_P(LogicalLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(LogicalLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(LogicalLayerTest_SW_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(LogicalLayerTest_HW_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},
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
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapes)), ::testing::ValuesIn(logicalOpTypes),
        ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputsPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto LogicalTestParamsNot = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapesNot)),
        ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_NOT),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputsPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs, LogicalLayerTest_NPU3700, LogicalTestParams,
                        LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNot, LogicalLayerTest_NPU3700, LogicalTestParamsNot,
                         LogicalLayerTest::getTestCaseName);

//
// NPU3720
//
std::set<ngraph::helpers::LogicalTypes> supportedTypes = {
        ngraph::helpers::LogicalTypes::LOGICAL_OR,
        ngraph::helpers::LogicalTypes::LOGICAL_XOR,
        ngraph::helpers::LogicalTypes::LOGICAL_AND,
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inShapes = {
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},   {{1, 16, 32}, {{1, 16, 32}}}, {{1, 28, 300, 1}, {{1, 1, 300, 28}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}}, {{2, 200}, {{2, 200}}},

};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> precommit_inShapes = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inShapesNot = {
        {{1, 2, 4}, {}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

std::vector<InferenceEngine::Precision> inputsPrecisionsNPU = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

std::vector<InferenceEngine::Precision> netPrecisionsNPU = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const auto logical_params = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inShapes)), ::testing::ValuesIn(supportedTypes),
        ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(netPrecisionsNPU),
        ::testing::ValuesIn(inputsPrecisionsNPU), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto precommit_logical_params = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(precommit_inShapes)), ::testing::ValuesIn(supportedTypes),
        ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(netPrecisionsNPU),
        ::testing::ValuesIn(inputsPrecisionsNPU), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto precommit_logical_params_not = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(inShapesNot)),
        ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_NOT),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT), ::testing::ValuesIn(netPrecisionsNPU),
        ::testing::ValuesIn(inputsPrecisionsNPU), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto tiling_logical_params = ::testing::Combine(
        ::testing::ValuesIn(LogicalLayerTest::combineShapes(tiling_inShapes)),
        ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_OR), ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(netPrecisionsNPU), ::testing::ValuesIn(inputsPrecisionsNPU),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(additional_config));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_logical, LogicalLayerTest_SW_NPU3720, logical_params, LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_logical, LogicalLayerTest_SW_NPU3720, precommit_logical_params,
                        LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_logical_not, LogicalLayerTest_SW_NPU3720, precommit_logical_params_not,
                        LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_tiling, LogicalLayerTest_HW_NPU3720, tiling_logical_params,
                        LogicalLayerTest::getTestCaseName);

}  // namespace
