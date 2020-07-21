// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

InferenceEngine::SizeVector axes = {0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {{{10, 10, 10, 10}}, {{10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<std::vector<std::vector<size_t>>> reshapeTargetShapes = {
    {{20, 10, 10, 10}, {20, 10, 10, 10}, {20, 10, 10, 10}},
    {{20, 20, 10, 10}, {20, 20, 10, 10}, {20, 20, 10, 10}},
    {{20, 20, 20, 10}, {20, 20, 20, 10}, {20, 20, 20, 10}},
    {{20, 20, 20, 20}, {20, 20, 20, 20}, {20, 20, 20, 20}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(DISABLED_NoReshape, ConcatLayerTest,
    ::testing::Combine(::testing::ValuesIn(axes), ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    ConcatLayerTest::getTestCaseName);
}  // namespace