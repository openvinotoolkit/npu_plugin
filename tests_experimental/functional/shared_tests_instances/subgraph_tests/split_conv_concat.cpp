// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_conv_concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8, InferenceEngine::Precision::I8};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(NoReshape, SplitConvConcat,
    ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 6, 40, 40})),
        ::testing::Values(InferenceEngine::SizeVector()), ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    SplitConvConcat::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Reshape, SplitConvConcat,
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::SizeVector({1, 6, 40, 40})),
        ::testing::ValuesIn({InferenceEngine::SizeVector({2, 6, 40, 40}), InferenceEngine::SizeVector({1, 6, 50, 50}),
            InferenceEngine::SizeVector({2, 6, 50, 50})}),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    SplitConvConcat::getTestCaseName);

}  // namespace
