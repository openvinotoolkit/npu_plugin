//
// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include <vpux/vpux_compiler_config.hpp>
#include <vpux/vpux_plugin_config.hpp>
#include "behavior/plugin/configuration_tests.hpp"
#include "ie_plugin_config.hpp"
#include "vpu_test_env_cfg.hpp"
#include "vpux/al/config/common.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> Configs = {
        {{VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}},
        {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}},
        {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}, {VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}},
        {{CONFIG_KEY(DEVICE_ID), ""}, {VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}},
        {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "1"}, {VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}},
        {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)},
         {VPUX_CONFIG_KEY(COMPILER_TYPE), VPUX_CONFIG_VALUE(DRIVER)}}};

const std::vector<std::map<std::string, std::string>> InConfigs = {
        {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
        {{CONFIG_KEY(PERF_COUNT), "YEP"}},
        {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},
        {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "TWENTY"}},
        {{CONFIG_KEY(PERFORMANCE_HINT), "IMPRESS_ME"}}};

static std::string getTestCaseName(testing::TestParamInfo<CorrectConfigParams> obj) {
    return CorrectConfigTests::getTestCaseName(obj) +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(CommonTestUtils::DEVICE_KEEMBAY);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(Configs)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(InConfigs)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(InConfigs)),
                         getTestCaseName);

}  // namespace
