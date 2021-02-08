// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/ctc_greedy_decoder.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbCTCGreedyDecoderLayerTest: public CTCGreedyDecoderLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeImport() override {
        throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
    }
};

TEST_P(KmbCTCGreedyDecoderLayerTest, CompareWithRefs) {
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

const std::vector<InferenceEngine::SizeVector> inputShapes = {
    InferenceEngine::SizeVector { 88, 1, 71 },
    InferenceEngine::SizeVector { 10, 1, 16 },
    InferenceEngine::SizeVector { 5, 4, 3 },
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

// TODO: [Track number: C#40001]
INSTANTIATE_TEST_CASE_P(
    DISABLED_CTCGreedyDecoder,
    KmbCTCGreedyDecoderLayerTest,
    params,
    CTCGreedyDecoderLayerTest::getTestCaseName
);

}  // namespace
