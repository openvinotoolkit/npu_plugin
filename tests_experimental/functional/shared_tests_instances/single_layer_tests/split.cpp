// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8, InferenceEngine::Precision::I8};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(NumSplitsCheck, SplitLayerTest,
    ::testing::Combine(::testing::Values(1), ::testing::Values(0, 1, 2, 3), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({30, 30, 30, 30})),
        ::testing::Values(InferenceEngine::SizeVector()), ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(REshape, SplitLayerTest,
    ::testing::Combine(::testing::Values(1), ::testing::Values(0), ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::SizeVector({30, 30, 30, 30})),
        ::testing::Values(InferenceEngine::SizeVector({40, 50, 60, 70})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    SplitLayerTest::getTestCaseName);

}  // namespace
