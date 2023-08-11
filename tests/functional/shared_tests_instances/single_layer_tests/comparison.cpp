//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"
#include <vector>
#include "common/functions.h"
#include "kmb_layer_test.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class VPUXComparisonLayerTest : public ComparisonLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
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

        auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first});

        auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
        if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondInput));
        }

        auto comparisonNode = ngraph::builder::makeComparison(inputs[0], secondInput, comparisonOpType);
        auto convertedComparisonNode = std::make_shared<ngraph::opset5::Convert>(comparisonNode, ngInputsPrc);
        function = std::make_shared<ngraph::Function>(convertedComparisonNode, inputs, "Comparison");
    }
};

class VPUXComparisonLayerTest_VPU3700 : public VPUXComparisonLayerTest {};

class VPUXComparisonLayerTest_SW_VPU3720 : public VPUXComparisonLayerTest {};

class VPUXComparisonLayerTest_HW_VPU3720 : public VPUXComparisonLayerTest {};

TEST_P(VPUXComparisonLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXComparisonLayerTest_SW_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXComparisonLayerTest_HW_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;
namespace {

// Shapes with more than 4 dimensions are not supported
std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{5}, {{1}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},

        // The same as for eltwise:
        // [Track number: E#15146]
        // Initialization disabled partly
        // {{2, 17, 3, 4}, {{4}, {1, 3, 4}}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

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

const auto ComparisonTestParams_MLIR = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_CompareWithRefs, VPUXComparisonLayerTest_VPU3700, ComparisonTestParams_MLIR,
                        VPUXComparisonLayerTest_VPU3700::getTestCaseName);

//
// VPU3720 Instantiation
//

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inShapesVPUX = {
        {{5}, {{1}}},
        {{10, 1}, {{1, 50}}},
        {{1, 16, 32}, {{1, 16, 32}}},
};

std::vector<InferenceEngine::Precision> precision_VPUX = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const auto comparison_params_VPUX = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inShapesVPUX)), ::testing::ValuesIn(precision_VPUX),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> precommit_inShapesVPUX = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

const auto precommit_comparison_params_VPUX = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(precommit_inShapesVPUX)),
        ::testing::ValuesIn(precision_VPUX), ::testing::ValuesIn(comparisonOpTypes_MLIR),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_Comparison_VPU3720, VPUXComparisonLayerTest_SW_VPU3720, comparison_params_VPUX,
                        VPUXComparisonLayerTest_SW_VPU3720::getTestCaseName);

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> tiling_inShapesVPU3720 = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

const auto tiling_comparison_params_VPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(tiling_inShapesVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(ngraph::helpers::ComparisonTypes::EQUAL),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_tiling_Comparison_VPU3720, VPUXComparisonLayerTest_HW_VPU3720,
                        tiling_comparison_params_VPU3720, VPUXComparisonLayerTest_HW_VPU3720::getTestCaseName);

}  // namespace
