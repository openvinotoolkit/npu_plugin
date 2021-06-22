// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/ctc_greedy_decoder_seq_len.hpp"
#include "kmb_layer_test.hpp"

#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>

using namespace ngraph;
namespace LayerTestsDefinitions {

class KmbCTCGreedyDecoderSeqLenLayerTest
        : public CTCGreedyDecoderSeqLenLayerTest,
          virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException(
                "MemoryAllocator:ProgrammableOutput - ArgumentError: "
                "ImplicitOutput_1_conversion:0::paddedShape[0] 6 - "
                "Does not match the dimension 1 of the tensor "
                "ImplicitOutput_1 already allocated in the given buffer");
        }
    }
    void SkipBeforeInfer() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("failing compilation");
        } else {
            throw LayerTestsUtils::KmbSkipTestException("differs from the reference");
        }
    }
};

TEST_P(KmbCTCGreedyDecoderSeqLenLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbCTCGreedyDecoderSeqLenLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> probPrecisions = {
    InferenceEngine::Precision::FP16
};
const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32
};

std::vector<bool> mergeRepeated{true, false};

const auto inputShape = std::vector<std::vector<size_t>>{
    {1, 1, 1}, {4, 80, 80}, {80, 4, 80}, {80, 80, 4}, {8, 20, 128}
};

const auto sequenceLengths = std::vector<int>{
        1, 50, 100
};

const auto blankIndexes = std::vector<int>{
        0, 50
};

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLenTests, KmbCTCGreedyDecoderSeqLenLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(inputShape),
            ::testing::ValuesIn(sequenceLengths),
            ::testing::ValuesIn(probPrecisions),
            ::testing::ValuesIn(idxPrecisions),
            ::testing::ValuesIn(blankIndexes),
            ::testing::ValuesIn(mergeRepeated),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
