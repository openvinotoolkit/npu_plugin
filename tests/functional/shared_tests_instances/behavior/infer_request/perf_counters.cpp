//
// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/infer_request/perf_counters.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

const std::vector<std::map<std::string, std::string>> configs = {
        {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(THROUGHPUT)}},
        {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(LATENCY)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPerfCountersTest::getTestCaseName);

}  // namespace
