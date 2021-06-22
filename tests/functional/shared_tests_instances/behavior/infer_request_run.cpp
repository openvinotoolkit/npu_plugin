// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_run.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, InferRequestRunTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                    ::testing::ValuesIn(configs)),
            InferRequestRunTests::getTestCaseName);

}  // namespace
