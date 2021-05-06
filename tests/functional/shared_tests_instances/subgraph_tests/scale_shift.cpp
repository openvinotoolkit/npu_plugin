// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/subgraph/scaleshift.hpp>

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace SubgraphTestsDefinitions {

class KmbScaleShiftLayerTest: public ScaleShiftLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::vector<std::vector<size_t>> inShape;
        std::tie(inShape, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();
        // There are errors for some input shapes with using of MCM-compiler:
        // [Debug  ][VPU][KMB nGraph Parser] Run MCM Compiler
        // [Error  ][VPU][KMB nGraph Parser] MCM Compiler exception: Shape:{100} - ArgumentError: index subscript 1 -
        // Exceeds the dimensionality 1
        // kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:156: Failure
        // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration) doesn't throw an exception.
        //  Actual: it throws:Shape:{100} - ArgumentError: index subscript 1 - Exceeds the dimensionality 1
        // [Track number: E#11548]
        if (isCompilerMCM()) {
            std::set<std::vector<std::vector<size_t>>> badShapesForMcm = {
                {{100}, {100}},
                {{4, 64}, {64}}
            };

            if (badShapesForMcm.find(inShape) != badShapesForMcm.end() ) {
                throw LayerTestsUtils::KmbSkipTestException("Bad shape: - ArgumentError: index subscript 1 - "
                                                            "Exceeds the dimensionality 1");
            }
        }
    }

    void SkipBeforeInfer() override {
        std::vector<std::vector<size_t>> inShape;
        std::tie(inShape, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();
        // Some input shapes with using of MLIR compiler lead to hang the board during infer step:
        // Last test output:
        // KmbLayerTestsCommon::Infer()
        // [Debug  ][VPU][VpualCoreNNExecutor] Allocated buffer for input with the size:
        // [Debug  ][VPU][VpualCoreNNExecutor] Allocated buffer for output with the size: 0
        // [Warning][VPU][InferRequest] SIPP/M2I is enabled but configuration is not supported.
        // [Info   ][VPU][VpualCoreNNExecutor] ::push started
        // [Warning][VPU][VpualCoreNNExecutor] Input blob is located in non-shareable memory. Need to do re-allocation.
        // [Info   ][VPU][VpualCoreNNExecutor] ::push finished
        // [Info   ][VPU][VpualCoreNNExecutor] pull started
        // [Track number: E#11546]
        if (isCompilerMLIR()) {
            std::set<std::vector<std::vector<size_t>>> badShapesForMLIR = {
                {{100}},
                {{200}},
                {{100}, {100}},
            };
            if (badShapesForMLIR.find(inShape) != badShapesForMLIR.end() ) {
                throw LayerTestsUtils::KmbSkipTestException("Infer hangs the board.");
            }
        }
    }

    void SkipBeforeValidate() override {
        std::vector<std::vector<size_t>> inShape;
        std::tie(inShape, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();
        // There are errors on validation step for some input shapes with using of MLIR compiler:
        // KmbLayerTestsCommon::Validate()
        // LayerTestsCommon::Validate()
        // openvino/inference-engine/tests/functional/shared_test_classes/
        // include/shared_test_classes/base/layer_test_utils.hpp:173: Failure
        // Value of: max != 0 && (diff <= static_cast<float>(threshold))
        //  Actual: false
        // Expected: true
        // Relative comparison of values expected: -24 and actual: 0 at index 2048
        // with threshold 0.0099999997764825821 failed TestReportProgress: KmbScaleShiftLayerTest validated
        // [Track number: E#11542]
        if (isCompilerMLIR()) {
            std::set<std::vector<std::vector<size_t>>> badShapesForMLIR = {
                    {{4, 64}, {64}},
                    {{1, 8}},
                    {{2, 16}},
                    {{3, 32}},
                    {{4, 64}},
                    {{5, 128}},
                    {{6, 256}},
                    {{7, 512}},
                    {{8, 1024}}
            };
            if (badShapesForMLIR.find(inShape) != badShapesForMLIR.end() ) {
                throw LayerTestsUtils::KmbSkipTestException("Comparison fails for this input shape");
            }
        }
        // There are errors on validation step for some input shapes with using of MCM compiler:
        // KmbLayerTestsCommon::Validate()
        // LayerTestsCommon::Validate()
        // openvino/inference-engine/tests/functional/shared_test_classes/include/
        // shared_test_classes/base/layer_test_utils.hpp:173: Failure
        // Value of: max != 0 && (diff <= static_cast<float>(threshold))
        //  Actual: false
        // Expected: true
        // Relative comparison of values expected: -15 and actual: 0.00019502639770507812 at index 1024
        // with threshold 0.0099999997764825821 failed TestReportProgress: KmbScaleShiftLayerTest validated
        // [Track number: E#11537]
        if (isCompilerMCM()) {
            std::set<std::vector<std::vector<size_t>>> badShapesForMCM = {
                {{2, 16}},
                {{3, 32}},
                {{4, 64}},
                {{5, 128}},
                {{6, 256}},
                {{7, 512}},
                {{8, 1024}}
            };
            if (badShapesForMCM.find(inShape) != badShapesForMCM.end() ) {
                throw LayerTestsUtils::KmbSkipTestException("Comparison fails for this input shape");
            }
        }

    }
};

TEST_P(KmbScaleShiftLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbScaleShiftLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace SubgraphTestsDefinitions

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{100}},
    {{200}},
    {{100}, {100}},
    {{4, 64}, {64}},
    {{1, 8}},
    {{2, 16}},
    {{3, 32}},
    {{4, 64}},
    {{5, 128}},
    {{6, 256}},
    {{7, 512}},
    {{8, 1024}},
    {{1, 8, 4, 4},     {1, 8, 1, 1}},
    {{1, 128, 32, 32}, {1, 128, 1, 1}},
    {{1, 512, 64, 64}, {1, 512, 1, 1}},
    {{1, 111, 3, 3},   {1, 111, 1, 1}},
};

std::vector<std::vector<float>> Scales = {
        {3.0f},
        {2.0f},
        {-1.0f},
        {-3.0f}
};

std::vector<std::vector<float>> Shifts = {
        {3.0f},
        {1.0f},
        {-1.0f},
        {-3.0f}
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_scale_shift, KmbScaleShiftLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(Scales),
                            ::testing::ValuesIn(Shifts)),
                        KmbScaleShiftLayerTest::getTestCaseName);

}  // namespace
