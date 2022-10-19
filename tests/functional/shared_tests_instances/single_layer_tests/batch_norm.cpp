//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/batch_norm.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbBatchNormLayerTest_VPU4000: public BatchNormLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Missing M2I runtime support");
    }
};

TEST_P(KmbBatchNormLayerTest_VPU4000, CompareWithRefs_MLIR_VPU4000) {
    useCompilerMLIR();
    setPlatformVPU4000();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// This test would only work if disabling BatchNormDecomposition
// from IE.cpp, 'addCommonOptimizationsPasses' func
// "pass_config->disable<ngraph::pass::BatchNormDecomposition>();"
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_BatchNorm_M2I,
        KmbBatchNormLayerTest_VPU4000,
        testing::Combine(
           testing::Values(0.001),                            // epsilon
           testing::Values(InferenceEngine::Precision::FP16), // netPrc
           testing::Values(InferenceEngine::Precision::FP16), // inPrc
           testing::Values(InferenceEngine::Precision::FP16), // outPrc
           testing::Values(InferenceEngine::Layout::ANY),     // inLayout
           testing::Values(InferenceEngine::Layout::ANY),     // outLayout
           testing::Values(InferenceEngine::SizeVector {1, 3, 256, 256}),
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbBatchNormLayerTest_VPU4000::getTestCaseName);

}  // namespace
