//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/comparison.hpp"
#include <vector>
#include "common/functions.h"
#include "ov_models/builders.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ComparisonLayerTestCommon : public ComparisonLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        ComparisonParams::InputShapesTuple inputShapes;
        InferenceEngine::Precision ngInputsPrecision;
        ngraph::helpers::ComparisonTypes comparisonOpType;
        ngraph::helpers::InputLayerType secondInputType;
        InferenceEngine::Precision ieInPrecision;
        InferenceEngine::Precision ieOutPrecision;

        std::map<std::string, std::string> additional_config;
        std::tie(inputShapes, ngInputsPrecision, comparisonOpType, secondInputType, ieInPrecision, ieOutPrecision,
                 targetDevice, additional_config) = this->GetParam();

        auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(ngInputsPrecision);
        configuration.insert(additional_config.begin(), additional_config.end());

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first))};

        auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
        if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputs.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(secondInput));
        }

        auto comparisonNode = ngraph::builder::makeComparison(inputs[0], secondInput, comparisonOpType);
        auto convertedComparisonNode = std::make_shared<ov::op::v0::Convert>(comparisonNode, ngInputsPrc);
        function = std::make_shared<ngraph::Function>(convertedComparisonNode, inputs, "Comparison");
    }
};

class ComparisonLayerTest_NPU3700 : public ComparisonLayerTestCommon {};

class ComparisonLayerTest_NPU3720 : public ComparisonLayerTestCommon {};
class ComparisonLayerTest_Tiling_NPU3720 : public ComparisonLayerTestCommon {};

TEST_P(ComparisonLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ComparisonLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ComparisonLayerTest_Tiling_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;
namespace {

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes_MLIR = {
        ngraph::helpers::ComparisonTypes::EQUAL,      ngraph::helpers::ComparisonTypes::LESS,
        ngraph::helpers::ComparisonTypes::LESS_EQUAL, ngraph::helpers::ComparisonTypes::NOT_EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER,    ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::PARAMETER,
        ngraph::helpers::InputLayerType::CONSTANT,
};

std::map<std::string, std::string> additional_config = {};

//
// NPU3700 Instantiation
//
// Shapes with more than 4 dimensions are not supported
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{5}, {{1}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const auto ComparisonTestParams_MLIR = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(inputShapes)), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs, ComparisonLayerTest_NPU3700, ComparisonTestParams_MLIR,
                        ComparisonLayerTest_NPU3700::getTestCaseName);

//
// NPU3720 Instantiation
//

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inShapes = {
        {{5}, {{1}}},
        {{10, 1}, {{1, 50}}},
        {{1, 16, 32}, {{1, 16, 32}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> precommit_inShapes = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

std::vector<InferenceEngine::Precision> precision = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const auto comparison_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(inShapes)), ::testing::ValuesIn(precision),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto precommit_comparison_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(precommit_inShapes)), ::testing::ValuesIn(precision),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto tiling_comparison_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(tiling_inShapes)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(ngraph::helpers::ComparisonTypes::EQUAL),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_Comparison, ComparisonLayerTest_NPU3720, comparison_params,
                        ComparisonLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_Comparison, ComparisonLayerTest_NPU3720, precommit_comparison_params,
                        ComparisonLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_tiling_Comparison, ComparisonLayerTest_Tiling_NPU3720, tiling_comparison_params,
                        ComparisonLayerTest_Tiling_NPU3720::getTestCaseName);

}  // namespace
