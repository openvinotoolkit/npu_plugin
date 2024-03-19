// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/configuration_tests.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "vpux/al/config/common.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
INSTANTIATE_TEST_SUITE_P(
        smoke_Basic, DefaultConfigurationTest,
        ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                           ::testing::Values(DefaultParameter{
                                   InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS,
                                   InferenceEngine::Parameter{std::string{"1"}}})),
        DefaultConfigurationTest::getTestCaseName);

IE_SUPPRESS_DEPRECATED_START
auto inconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}};
};

auto multiinconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}}};
};

auto autoinconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}}};
};

auto auto_batch_inconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "-1"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}};
};

IE_SUPPRESS_DEPRECATED_END

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(inconfigs())),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiinconfigs())),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoinconfigs())),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         IncorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>> conf = {{}};

auto autoConfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              modelPriorityToString(ov::hint::Priority::HIGH)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              modelPriorityToString(ov::hint::Priority::MEDIUM)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, modelPriorityToString(ov::hint::Priority::LOW)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_NONE}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_ERROR}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_WARNING}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_INFO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_DEBUG}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_TRACE}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              modelPriorityToString(ov::hint::Priority::HIGH)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              modelPriorityToString(ov::hint::Priority::MEDIUM)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              modelPriorityToString(ov::hint::Priority::LOW)}}};
};

auto auto_batch_configs = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_NPU},
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}}};
};

namespace DefaultValuesConfigTestName {
static std::string getTestCaseName(testing::TestParamInfo<CorrectConfigParams> obj) {
    return CorrectConfigTests::getTestCaseName(obj) +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
}
}  // namespace DefaultValuesConfigTestName

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, DefaultValuesConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn(conf)),
                         DefaultValuesConfigTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(inconfigs())),
                         DefaultValuesConfigTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiinconfigs())),
                         DefaultValuesConfigTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoinconfigs())),
                         DefaultValuesConfigTestName::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         DefaultValuesConfigTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, CorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_configs())),
                         DefaultValuesConfigTestName::getTestCaseName);

const std::vector<std::map<std::string, std::string>> vpu_prop_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "8"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO},
}};

const std::vector<std::map<std::string, std::string>> vpu_loadNetWork_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
        //{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS,
        // InferenceEngine::PluginConfigParams::NO},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "10"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES},
}};

auto auto_multi_prop_config = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT},
             //{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS,
             // InferenceEngine::PluginConfigParams::YES},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "2"},
             {InferenceEngine::PluginConfigParams::KEY_ALLOW_AUTO_BATCHING, InferenceEngine::PluginConfigParams::NO},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}};
};

auto auto_multi_loadNetWork_config = []() {
    return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             //{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS,
             // InferenceEngine::PluginConfigParams::NO},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "10"},
             {InferenceEngine::PluginConfigParams::KEY_ALLOW_AUTO_BATCHING, InferenceEngine::PluginConfigParams::YES},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}};
};

namespace SetPropLoadNetWorkGetPropTestName {
static std::string getTestCaseName(testing::TestParamInfo<LoadNetWorkPropertiesParams> obj) {
    std::string target_device;
    std::map<std::string, std::string> configuration;
    std::map<std::string, std::string> loadNetWorkConfig;
    std::tie(target_device, configuration, loadNetWorkConfig) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
           << "_";
    if (!configuration.empty()) {
        result << "configItem=";
        for (auto& configItem : configuration) {
            result << configItem.first << "_" << configItem.second << "_";
        }
    }

    if (!loadNetWorkConfig.empty()) {
        result << "loadNetWorkConfig=";
        for (auto& configItem : loadNetWorkConfig) {
            result << configItem.first << "_" << configItem.second << "_";
        }
    }

    return result.str();
}
}  // namespace SetPropLoadNetWorkGetPropTestName

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, SetPropLoadNetWorkGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(vpu_prop_config),
                                            ::testing::ValuesIn(vpu_loadNetWork_config)),
                         SetPropLoadNetWorkGetPropTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, SetPropLoadNetWorkGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_prop_config()),
                                            ::testing::ValuesIn(auto_multi_loadNetWork_config())),
                         SetPropLoadNetWorkGetPropTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, SetPropLoadNetWorkGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_multi_prop_config()),
                                            ::testing::ValuesIn(auto_multi_loadNetWork_config())),
                         SetPropLoadNetWorkGetPropTestName::getTestCaseName);
}  // namespace
