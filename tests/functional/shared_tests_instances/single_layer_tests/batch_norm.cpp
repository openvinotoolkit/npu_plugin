//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/batch_norm.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXBatchNormLayerTest : public BatchNormLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXBatchNormLayerTest_VPU3720 : public VPUXBatchNormLayerTest {};

TEST_P(VPUXBatchNormLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<size_t>> inShapes_precommit = {
        std::vector<size_t>{1, 5, 20, 20},
};
const std::vector<std::vector<size_t>> inShapes = {
        std::vector<size_t>{6, 7},
        std::vector<size_t>{3, 3, 5},
        std::vector<size_t>{5, 7, 6, 3},
        std::vector<size_t>{1, 3, 256, 256},
};

const auto paramsConfig =
        testing::Combine(testing::Values(0.001),                             // epsilon
                         testing::Values(InferenceEngine::Precision::FP16),  // netPrc
                         testing::Values(InferenceEngine::Precision::FP16),  // inPrc
                         testing::Values(InferenceEngine::Precision::FP16),  // outPrc
                         testing::Values(InferenceEngine::Layout::ANY),      // inLayout
                         testing::Values(InferenceEngine::Layout::ANY),      // outLayout
                         testing::ValuesIn(inShapes), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsPrecommit = testing::Combine(testing::Values(0.001),                             // epsilon
                                              testing::Values(InferenceEngine::Precision::FP16),  // netPrc
                                              testing::Values(InferenceEngine::Precision::FP16),  // inPrc
                                              testing::Values(InferenceEngine::Precision::FP16),  // outPrc
                                              testing::Values(InferenceEngine::Layout::ANY),      // inLayout
                                              testing::Values(InferenceEngine::Layout::ANY),      // outLayout
                                              testing::ValuesIn(inShapes_precommit),
                                              testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchNorm, VPUXBatchNormLayerTest_VPU3720, paramsPrecommit,
                         VPUXBatchNormLayerTest_VPU3720::getTestCaseName);

}  // namespace
