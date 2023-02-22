//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/ctc_greedy_decoder_seq_len.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>

using namespace ngraph;
namespace LayerTestsDefinitions {

class KmbCTCGreedyDecoderSeqLenLayerTest :
        public CTCGreedyDecoderSeqLenLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
    }
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("differs from the reference");
    }
};

class KmbCTCGreedyDecoderSeqLenLayerTest_MLIR_VPU3720 :
        public CTCGreedyDecoderSeqLenLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbCTCGreedyDecoderSeqLenLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbCTCGreedyDecoderSeqLenLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbCTCGreedyDecoderSeqLenLayerTest_MLIR_VPU3720, COMPILER_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> probPrecisions = {InferenceEngine::Precision::FP16};
const std::vector<InferenceEngine::Precision> idxPrecisions = {InferenceEngine::Precision::I32};

std::vector<bool> mergeRepeated{true, false};

const auto inputShape =
        std::vector<std::vector<size_t>>{{1, 1, 1}, {4, 80, 80}, {80, 4, 80}, {80, 80, 4}, {8, 20, 128}};

const auto sequenceLengths = std::vector<int>{1, 50, 100};

const auto blankIndexes = std::vector<int>{0, 50};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_CTCGreedyDecoderSeqLenTests, KmbCTCGreedyDecoderSeqLenLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShape), ::testing::ValuesIn(sequenceLengths),
                                            ::testing::ValuesIn(probPrecisions), ::testing::ValuesIn(idxPrecisions),
                                            ::testing::ValuesIn(blankIndexes), ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);

const auto params_MLIR_VPU3720 = testing::Combine(
        ::testing::ValuesIn(inputShape), ::testing::ValuesIn(sequenceLengths), ::testing::ValuesIn(probPrecisions),
        ::testing::ValuesIn(idxPrecisions), ::testing::ValuesIn(blankIndexes), ::testing::ValuesIn(mergeRepeated),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLenTests_VPU3720, KmbCTCGreedyDecoderSeqLenLayerTest_MLIR_VPU3720,
                         params_MLIR_VPU3720, CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
