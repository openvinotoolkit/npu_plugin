// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/relu.hpp"

using namespace LayerTestsDefinitions;

namespace {
// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};
const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 50, 50})),
        ::testing::Values(InferenceEngine::SizeVector()),
        ::testing::Values(false),
        ::testing::Values("KMB")
);


const auto reshapeCases = ::testing::Combine(
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 50, 50})),
        ::testing::ValuesIn({InferenceEngine::SizeVector({1, 3, 100, 100}),
                             InferenceEngine::SizeVector({1, 3, 10, 10}),
                             InferenceEngine::SizeVector({2, 3, 50, 50}),
                            }),
        ::testing::Values(false, true),
        ::testing::Values("KMB")
);


INSTANTIATE_TEST_CASE_P(ReLu_Basic, ReLuLayerTest, basicCases, ReLuLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ReLu_Reshape, ReLuLayerTest, reshapeCases, ReLuLayerTest::getTestCaseName);

}