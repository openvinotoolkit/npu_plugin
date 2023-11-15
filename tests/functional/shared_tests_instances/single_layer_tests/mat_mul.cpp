//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "single_layer_tests/mat_mul.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXMatMulLayerTest : public MatMulTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXMatMulLayerTest_VPU3700 : public VPUXMatMulLayerTest {
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
            throw LayerTestsUtils::VpuSkipTestException("Unsupported MLIR case");
        }
    }
    void SkipBeforeInfer() override {
        // Tracking number [E#85137]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::VpuSkipTestException("AppendGraphInitialize result 0x70000001");
        }
    }
};
class VPUXMatMulLayerTest_HW_VPU3720 : public VPUXMatMulLayerTest {};
class VPUXMatMulLayerTest_SW_VPU3720 : public VPUXMatMulLayerTest {};

TEST_P(VPUXMatMulLayerTest_VPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulLayerTest_HW_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulLayerTest_SW_VPU3720, SW) {
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
        ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(additional_config));

const auto matMulParams = ::testing::Combine(
        ::testing::ValuesIn(shapeRelatedParams), ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(additional_config));

const std::vector<InferenceEngine::Precision> inputPrecisions_VPU3720 = {InferenceEngine::Precision::FP16};

const std::vector<ShapeRelatedParams> shapeRelatedParams_VPU3720 = {
        {{{1, 2, 5, 16}, false}, {{1, 2, 16, 4}, false}}, {{{1, 8, 76, 64}, false}, {{1, 8, 4, 64}, true}},
        {{{2, 16, 5}, true}, {{16, 16}, true}},           {{{8, 76, 64}, false}, {{4, 64}, true}},
        {{{1, 16, 16, 2}, false}, {{1, 2, 2}, false}},    {{{8, 64, 76}, true}, {{64, 4}, false}},
        {{{1, 1, 1, 3}, false}, {{12, 3}, true}}};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes_VPU3720 = {
        ngraph::helpers::InputLayerType::PARAMETER,
};

const auto matMulPrecommit = ::testing::Combine(
        ::testing::ValuesIn(shapeRelatedParams_VPU3720), ::testing::ValuesIn(inputPrecisions_VPU3720),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(secondaryInputTypes_VPU3720),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

/* ============= VPU3700 ============= */

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_MatMul, VPUXMatMulLayerTest_VPU3700, matMulParams,
                         VPUXMatMulLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_MatMul_to_FC_case, VPUXMatMulLayerTest_VPU3700, fullyConnectedCase,
                         VPUXMatMulLayerTest_VPU3700::getTestCaseName);

/* ============= VPU3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_MatMul, VPUXMatMulLayerTest_HW_VPU3720, matMulPrecommit,
                         VPUXMatMulLayerTest_HW_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_to_FC_case, VPUXMatMulLayerTest_SW_VPU3720, fullyConnectedCase,
                         VPUXMatMulLayerTest_SW_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul, VPUXMatMulLayerTest_SW_VPU3720, matMulParams,
                         VPUXMatMulLayerTest_SW_VPU3720::getTestCaseName);

}  // namespace
