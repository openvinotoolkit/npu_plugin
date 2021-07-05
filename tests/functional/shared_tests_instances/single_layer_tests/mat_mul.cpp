// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/mat_mul.hpp"

namespace LayerTestsDefinitions {

class KmbMatMulLayerTest : public MatMulTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout;
        ShapeRelatedParams shapeRelatedParams;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::string targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice,
                 additionalConfig) = GetParam();

        if (isCompilerMCM()) {
            if (shapeRelatedParams.input1.first == InferenceEngine::SizeVector({1, 16})) {
                throw LayerTestsUtils::KmbSkipTestException("Unsupported MCM case");
            }
        } else {
            if (shapeRelatedParams.input1.first == InferenceEngine::SizeVector({1, 2048})) {
                throw LayerTestsUtils::KmbSkipTestException("Unsupported MLIR case");
            }
        }
    }
    void SkipBeforeValidate() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    }
};

TEST_P(KmbMatMulLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbMatMulLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbMatMulLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32,
                                                                 InferenceEngine::Precision::FP16};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {{{{1, 4, 5, 6}, false}, {{1, 4, 6, 4}, false}},
                                                            {{{4, 5, 6}, false}, {{6, 3}, false}},
                                                            {{{9, 9, 9}, false}, {{9, 9}, false}}};

const std::vector<ShapeRelatedParams> fullyConnectedShapeParams = {{{{1, 16}, false}, {{64, 16}, true}}};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

const std::vector<ShapeRelatedParams> shapeRelatedParams_kpi_mcm = {
        {{{1, 2048}, false}, {{1000, 2048}, true}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes_kpi_mcm = {
        ngraph::helpers::InputLayerType::CONSTANT,
};

std::map<std::string, std::string> additional_config = {};

const auto fullyConnectedCase = ::testing::Combine(
        ::testing::ValuesIn(fullyConnectedShapeParams), ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

// Test is disabled due to two types of errors:
// 1. On step [Debug  ][VPU][KMB nGraph Parser] Convert nGraph to MCM Model
// kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:169: Failure
// Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
// doesn't throw an exception.
// Actual: it throws:Unsupported operation: MatMul_1827 with name MatMul_2189 with type MatMul with C++
// type N6ngraph2op2v06MatMulE
// kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1880
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:66
//
// 2. On step [Debug  ][VPU][KMB nGraph Parser] Convert nGraph to MCM Model
// ERROR:   Op:McmFC_2576 - OpError: Invalid input data (0) - Inconsistent total size of
// input tensor (input 0) 120 and 1st dimension of weights tensor (input 1) 6
// Segmentation fault (core dumped)
// [Track number: S#50186]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_MatMul, KmbMatMulLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapeRelatedParams),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(secondaryInputTypes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                           ::testing::Values(additional_config)),
                        KmbMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MatMul_to_FC_case, KmbMatMulLayerTest, fullyConnectedCase,
                        KmbMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_MatMul_kpi_mcm, KmbMatMulLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapeRelatedParams_kpi_mcm),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(secondaryInputTypes_kpi_mcm),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                           ::testing::Values(additional_config)),
                        KmbMatMulLayerTest::getTestCaseName);

}  // namespace
