// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/ctc_greedy_decoder.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbCTCGreedyDecoderLayerTest:
        public CTCGreedyDecoderLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

class KmbCTCGreedyDecoderLayerTest_MLIR : public KmbCTCGreedyDecoderLayerTest {
    void SkipBeforeLoad() override {
        // Input shapes below were gotten from CPU-plugin test. They produce 2 error types for MLIR-compiler:
        // 1) Shapes beginning with {50, ..} lead to error at compile step:
        // KmbLayerTestsCommon::Compile
        // [Debug  ][VPU][VPUXBackends] Searching for device VPU-0 to use started...
        // [Warning][VPU][VPUXBackends] Device to use not found!
        // [vpux-compiler] Failed Pass LowerIERT2VPUIP on Operation loc(unknown)
        // [vpux-compiler] Got Diagnostic at loc("CTCGreedyDecoder_6570") : Input tensor [T N C] = [50 3 55]
        // has unsupported dimension size N != 1 loc("CTCGreedyDecoder_6570"):
        // error: Input tensor [T N C] = [50 3 55] has unsupported dimension size N != 1
        // [vpux-compiler] Failed Pass mlir::detail::OpToOpPassAdaptor on Operation loc(unknown)
        //
        // 2) Shape { 1, 1, 16 } lead to error on kmb-board:
        // [Error  ][VPU][VpualCoreNNExecutor] allocateGraph: failed to create NnCorePlg
        // kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:165: Failure
        // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
        // doesn't throw an exception.
        // Actual: it throws:VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: 6
        // [Track number: E#12272]
        const std::vector<InferenceEngine::SizeVector> badInputShapesForMLIR = {
                InferenceEngine::SizeVector {50, 3, 3},
                InferenceEngine::SizeVector {50, 3, 128},
                InferenceEngine::SizeVector {1, 1, 16}
        };

        InferenceEngine::SizeVector inShape;
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                 inShape, std::ignore, std::ignore) = GetParam();
        for (auto iter = badInputShapesForMLIR.cbegin(); iter != badInputShapesForMLIR.cend(); iter++) {
            if (inShape == *iter) {
                throw LayerTestsUtils::KmbSkipTestException("Input tensor [T N C] has unsupported "
                                                            "dimension size N != 1");
            }
        }
    }
};

class KmbCTCGreedyDecoderLayerTest_MCM : public KmbCTCGreedyDecoderLayerTest {};

TEST_P(KmbCTCGreedyDecoderLayerTest_MCM, CompareWithRefs) {
    Run();
}

TEST_P(KmbCTCGreedyDecoderLayerTest_MLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<bool> mergeRepeated = {true, false};

// Only batch = 1 is supported
const std::vector<InferenceEngine::SizeVector> inputShapes_MLIR = {
    InferenceEngine::SizeVector {88, 1, 71},
    InferenceEngine::SizeVector {10, 1, 16},
    InferenceEngine::SizeVector {50, 3, 3},
    InferenceEngine::SizeVector {50, 3, 128},
    InferenceEngine::SizeVector {1, 1, 16}
};

const std::vector<InferenceEngine::SizeVector> inputShapes_MCM =  {
    InferenceEngine::SizeVector {88, 1, 71},
    InferenceEngine::SizeVector {10, 1, 16}
};

const auto params_MLIR = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes_MLIR),
    testing::ValuesIn(mergeRepeated),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto params_MCM = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes_MCM),
        testing::ValuesIn(mergeRepeated),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);


INSTANTIATE_TEST_CASE_P(
    smoke_CTCGreedyDecoder,
    KmbCTCGreedyDecoderLayerTest_MLIR,
    params_MLIR,
    CTCGreedyDecoderLayerTest::getTestCaseName
);

INSTANTIATE_TEST_CASE_P(
        smoke_CTCGreedyDecoder,
        KmbCTCGreedyDecoderLayerTest_MCM,
        params_MCM,
        CTCGreedyDecoderLayerTest::getTestCaseName
);


}  // namespace
