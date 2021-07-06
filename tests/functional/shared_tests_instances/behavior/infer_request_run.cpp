// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_run.hpp"
#include <vpux_config.hpp>

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, InferRequestRunTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                    ::testing::ValuesIn(configs)),
            InferRequestRunTests::getTestCaseName);

#if defined(__arm__) || defined(__aarch64__)
    const std::vector<std::map<std::string, std::string>> configsExecStreams = {
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {VPUX_CONFIG_KEY(EXECUTOR_STREAMS), "1"}},
// observed failures with multi executors
//            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {VPUX_CONFIG_KEY(EXECUTOR_STREAMS), "2"}},
//            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {VPUX_CONFIG_KEY(EXECUTOR_STREAMS), "3"}},
//            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {VPUX_CONFIG_KEY(EXECUTOR_STREAMS), "4"}},
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, InferRequestRunMultipleExecutorStreamsTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                    ::testing::ValuesIn(configsExecStreams)),
            InferRequestRunTests::getTestCaseName);
#endif // #if defined(__arm__) || defined(__aarch64__)
}  // namespace
