//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"
#include <vector>
#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class KmbComparisonLayerTest : public ComparisonLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class KmbComparisonLayerTestSetUp : public KmbComparisonLayerTest {
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

class KmbComparisonLayerTest_MLIR : public KmbComparisonLayerTestSetUp {};

class VPUXComparisonLayerTest_MLIR_SW_VPU3720 : public KmbComparisonLayerTestSetUp {};

class VPUXComparisonLayerTest_MLIR_HW_VPU3720 : public KmbComparisonLayerTestSetUp {};

TEST_P(KmbComparisonLayerTest_MLIR, CompareWithRefs) {
    useCompilerMLIR();
    Run();
}

TEST_P(VPUXComparisonLayerTest_MLIR_SW_VPU3720, CompareWithRefs) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXComparisonLayerTest_MLIR_HW_VPU3720, CompareWithRefs) {
    useCompilerMLIR();
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

std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};

// The operations are not supported for now for mcm compiler
std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
        ngraph::helpers::ComparisonTypes::EQUAL,   ngraph::helpers::ComparisonTypes::NOT_EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER, ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
        ngraph::helpers::ComparisonTypes::LESS,    ngraph::helpers::ComparisonTypes::LESS_EQUAL,
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

const auto ComparisonTestParams = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)), ::testing::ValuesIn(inputsPrecisions),
        ::testing::ValuesIn(comparisonOpTypes), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

const auto ComparisonTestParams_MLIR = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(comparisonOpTypes_MLIR), ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_CompareWithRefs, KmbComparisonLayerTest_MLIR, ComparisonTestParams_MLIR,
                        KmbComparisonLayerTest::getTestCaseName);

//
// VPU3720 Instantiation
//

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inShapesVPU3720 = {
        {{5}, {{1}}},
        {{1, 16, 32}, {{1, 16, 32}}},
};

const auto comparison_params_VPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inShapesVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::ValuesIn(comparisonOpTypes_MLIR),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_CompareWithRefs_VPU3720, VPUXComparisonLayerTest_MLIR_SW_VPU3720,
                        comparison_params_VPU3720, KmbComparisonLayerTest::getTestCaseName);

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> precommit_inShapesVPU3720 = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

const auto precommit_comparison_params_VPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(precommit_inShapesVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::ValuesIn(comparisonOpTypes_MLIR),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_CompareWithRefs_VPU3720, VPUXComparisonLayerTest_MLIR_SW_VPU3720,
                        precommit_comparison_params_VPU3720, KmbComparisonLayerTest::getTestCaseName);

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> tiling_inShapesVPU3720 = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

const auto tiling_comparison_params_VPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(tiling_inShapesVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(ngraph::helpers::ComparisonTypes::EQUAL),
        ::testing::ValuesIn(secondInputTypes), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_tiling_CompareWithRefs_VPU3720, VPUXComparisonLayerTest_MLIR_HW_VPU3720,
                        tiling_comparison_params_VPU3720, KmbComparisonLayerTest::getTestCaseName);

}  // namespace
