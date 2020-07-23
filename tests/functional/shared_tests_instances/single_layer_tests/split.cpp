// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(DISABLED_NumSplitsCheck, SplitLayerTest,
    ::testing::Combine(::testing::Values(1), ::testing::Values(0, 1, 2, 3),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({30, 30, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    SplitLayerTest::getTestCaseName);
}  // namespace
