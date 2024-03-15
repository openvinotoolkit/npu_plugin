// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <vector>
#include "behavior/plugin/configuration_tests.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ie_plugin_config.hpp"
#include "vpux/al/config/common.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> Configs = {
        {{ov::intel_vpux::compiler_type.name(), "DRIVER"}},
        {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}, {ov::intel_vpux::compiler_type.name(), "DRIVER"}},
        {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}, {ov::intel_vpux::compiler_type.name(), "DRIVER"}},
        {{CONFIG_KEY(DEVICE_ID), ""}, {ov::intel_vpux::compiler_type.name(), "DRIVER"}},
        {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "1"}, {ov::intel_vpux::compiler_type.name(), "DRIVER"}},
        {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}, {ov::intel_vpux::compiler_type.name(), "DRIVER"}}};

const std::vector<std::map<std::string, std::string>> InConfigs = {
        {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
        {{CONFIG_KEY(PERF_COUNT), "YEP"}},
        {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},
        {{CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS), "TWENTY"}},
        {{CONFIG_KEY(PERFORMANCE_HINT), "IMPRESS_ME"}}};

static std::string getTestCaseName(testing::TestParamInfo<CorrectConfigParams> obj) {
    return CorrectConfigTests::getTestCaseName(obj) +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(Configs)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(InConfigs)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(InConfigs)),
                         getTestCaseName);

}  // namespace
