// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbShuffleChannelsLayerTest: public ShuffleChannelsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

};

TEST_P(KmbShuffleChannelsLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {
     {3, 4, 9, 5}, {2, 16, 24, 15}, {1, 32, 12, 25}
};

const std::vector<std::tuple<int, int>> shuffleParameters = {
      std::make_tuple(1, 2), std::make_tuple(-3, 2),
      std::make_tuple(2, 3), std::make_tuple(-2, 3),
      std::make_tuple(3, 5), std::make_tuple(-1, 5)
};

const auto params = testing::Combine(
    testing::ValuesIn(shuffleParameters),
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        smoke_ShuffleChannels,
        KmbShuffleChannelsLayerTest,
        params,
        KmbShuffleChannelsLayerTest::getTestCaseName
);

}  // namespace



namespace { // conformance scenarios

const std::vector<std::vector<size_t>> inShapes = {
      {1,   4,  3,  2},
      {1, 116, 28, 28},
      {1, 232, 14, 14},
      {1, 464,  7,  7},
      {1,  32, 28, 28},
      {1,  64, 14, 14},
      {1, 128,  7,  7},
      {1,  24, 28, 28},
      {1,  48, 14, 14},
      {1,  96,  7,  7},
};

const std::vector<std::tuple<int, int>> shParams = {
      std::make_tuple(1, 2) // axis=1, group=2
};

const auto params2 = testing::Combine(
    testing::ValuesIn(shParams),
    testing::Values(InferenceEngine::Precision::FP16),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        conform_ShuffleChannels,
        KmbShuffleChannelsLayerTest,
        params2,
        KmbShuffleChannelsLayerTest::getTestCaseName
);

}  // namespace
