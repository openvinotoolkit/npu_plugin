// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <vpux/vpux_plugin_config.hpp>
#include <vpux/vpux_compiler_config.hpp>
#include "ie_plugin_config.hpp"
#include "behavior/plugin/configuration_tests.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> Configs = {
    {},
    {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
    {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}},
    {{CONFIG_KEY(DEVICE_ID), ""}},
    {{VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), "1"}},
    {{KMB_CONFIG_KEY(THROUGHPUT_STREAMS), "1"}},
    {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "1"}}
};

const std::vector<std::map<std::string, std::string>> InConfigs = {
    {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
    {{CONFIG_KEY(PERF_COUNT), "YEP"}},
    {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},
    {{VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), "TWENTY"}},
    {{KMB_CONFIG_KEY(THROUGHPUT_STREAMS), "TWENTY"}},
    {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "TWENTY"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
    ::testing::Combine(
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
    ::testing::ValuesIn(Configs)),
    CorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
    ::testing::Combine(
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
    ::testing::ValuesIn(InConfigs)),
    IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
    ::testing::Combine(
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
    ::testing::ValuesIn(InConfigs)),
    IncorrectConfigAPITests::getTestCaseName);

} // namespace
