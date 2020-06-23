// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/test_plugin.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecision = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTests,
                        ::testing::Combine(
                            ::testing::Values(InferenceEngine::Precision::FP32),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        BehaviorTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTestInput,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        BehaviorTestInput::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTestOutput,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecision),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        BehaviorTestOutput::getTestCaseName);
}  // namespace
