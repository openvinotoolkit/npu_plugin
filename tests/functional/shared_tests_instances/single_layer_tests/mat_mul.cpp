// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
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

        if (shapeRelatedParams.input1.first == InferenceEngine::SizeVector({1, 2048})) {
            throw LayerTestsUtils::KmbSkipTestException("Unsupported MLIR case");
        }
    }
    void SkipBeforeInfer() override {
        // [Track number: E#20337]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::KmbSkipTestException("AppendGraphInitialize result 0x70000001");
        }
    }
    void SkipBeforeValidate() override {
    }
};
class VPUXMatMulLayerTest_HW : public MatMulTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXMatMulLayerTest_SW : public MatMulTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

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
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulLayerTest_HW, HW_MLIR_VPU3720) {
    setPlatformVPU3720();
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulLayerTest_SW, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
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

const std::vector<ShapeRelatedParams> fullyConnectedShapeParams = {
        {{{1, 16}, false}, {{64, 16}, true}},
        {{{2, 16}, false}, {{64, 16}, true}},
        {{{1, 16}, false}, {{16, 64}, false}},
        {{{2, 1, 512}, false}, {{2, 40, 512}, true}},
        {{{1, 8, 4, 64}, false}, {{1, 8, 64, 76}, false}},
        {{{1, 1, 256}, false}, {{1, 16, 256}, true}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
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
// vpux-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:169: Failure
// Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
// doesn't throw an exception.
// Actual: it throws:Unsupported operation: MatMul_1827 with name MatMul_2189 with type MatMul with C++
// type N6ngraph2op2v06MatMulE
// vpux-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1880
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:66
//
// 2. On step [Debug  ][VPU][KMB nGraph Parser] Convert nGraph to MCM Model
// ERROR:   Op:McmFC_2576 - OpError: Invalid input data (0) - Inconsistent total size of
// input tensor (input 0) 120 and 1st dimension of weights tensor (input 1) 6
// Segmentation fault (core dumped)
// [Track number: S#50186]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_MatMul, KmbMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapeRelatedParams),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(secondaryInputTypes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         KmbMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_MatMul_to_FC_case, KmbMatMulLayerTest, fullyConnectedCase,
                         KmbMatMulLayerTest::getTestCaseName);

/* ============= VPU3720 ============= */

const std::vector<InferenceEngine::Precision> inputPrecisions_VPU3720 = {InferenceEngine::Precision::FP16};
const std::vector<ShapeRelatedParams> shapeRelatedParams_VPU3720 = {{{{1, 2, 5, 16}, false}, {{1, 2, 16, 4}, false}},
                                                                    {{{1, 8, 76, 64}, false}, {{1, 8, 4, 64}, true}},
                                                                    {{{2, 16, 5}, true}, {{16, 16}, true}},
                                                                    {{{8, 76, 64}, false}, {{4, 64}, true}},
                                                                    {{{8, 64, 76}, true}, {{64, 4}, false}}};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes_VPU3720 = {
        ngraph::helpers::InputLayerType::PARAMETER,
};

const auto params_VPU3720 = ::testing::Combine(
        ::testing::ValuesIn(shapeRelatedParams_VPU3720), ::testing::ValuesIn(inputPrecisions_VPU3720),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(secondaryInputTypes_VPU3720), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_MatMul, VPUXMatMulLayerTest_HW, params_VPU3720,
                         KmbMatMulLayerTest::getTestCaseName);

}  // namespace
