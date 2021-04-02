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

TEST_P(KmbCTCGreedyDecoderLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbCTCGreedyDecoderLayerTest, CompareWithRefs_MLIR) {
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
const std::vector<InferenceEngine::SizeVector> inputShapes = {
    InferenceEngine::SizeVector { 88, 1, 71 },
    InferenceEngine::SizeVector { 10, 1, 16 },
};

const auto params = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes),
    testing::ValuesIn(mergeRepeated),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
    smoke_CTCGreedyDecoder,
    KmbCTCGreedyDecoderLayerTest,
    params,
    CTCGreedyDecoderLayerTest::getTestCaseName
);

}  // namespace
