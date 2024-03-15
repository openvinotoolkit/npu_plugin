// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request/config.hpp"
#include <vector>
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ie_plugin_config.hpp"
#include "overload/infer_request/config.hpp"

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
        {{ov::hint::model_priority.name(), modelPriorityToString(ov::hint::Priority::HIGH)}},
        {{ov::hint::enable_cpu_pinning.name(), CONFIG_VALUE(YES)}},

        // Private options
        {{"NPU_PLATFORM", "AUTO_DETECT"}},
};

const std::vector<std::map<std::string, std::string>> Inconfigs = {
        // Public options
        {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
        {{CONFIG_KEY(PERF_COUNT), "YEP"}},
        {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},

        // Private options
        {{"NPU_PLATFORM", "SOME_PLATFORM"}},
};

static std::string getTestCaseName(testing::TestParamInfo<BehaviorTestsDefinitions::InferRequestParams> obj) {
    return InferRequestConfigTest::getTestCaseName(obj) +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestConfigTest,
                         ::testing::Combine(::testing::Values(2u), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         getTestCaseName);
// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestConfigTestVpux,
                         ::testing::Combine(::testing::Values(2u), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         getTestCaseName);
}  // namespace
