// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_conv_concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_NoReshape, SplitConvConcat,
    ::testing::Combine(::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 6, 40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    SplitConvConcat::getTestCaseName);
}  // namespace
