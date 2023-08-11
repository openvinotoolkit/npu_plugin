// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"
#include <vector>
#include <vpux/vpux_plugin_config.hpp>
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
        // Public options
        {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
        {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}, {CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT)}},
        {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}, {CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}},
        {{CONFIG_KEY(DEVICE_ID), ""}},
        {{CONFIG_KEY(MODEL_PRIORITY), CONFIG_VALUE(MODEL_PRIORITY_HIGH)}},

        // Private options
        {{"VPUX_PLATFORM", "AUTO_DETECT"}},
};

const std::vector<std::map<std::string, std::string>> Inconfigs = {
        // Public options
        {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
        {{CONFIG_KEY(PERF_COUNT), "YEP"}},
        {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},

        // Private options
        {{"VPUX_PLATFORM", "SOME_PLATFORM"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestConfigTest,
                         ::testing::Combine(::testing::Values(2u), ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestConfigTest::getTestCaseName);
}  // namespace
