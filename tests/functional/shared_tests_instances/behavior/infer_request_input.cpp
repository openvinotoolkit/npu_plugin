// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request_input.hpp"
#include "ie_plugin_config.hpp"

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8
};

const std::vector<std::map<std::string, std::string>> configs = {};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestInputTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        InferRequestInputTests::getTestCaseName);
}  // namespace
