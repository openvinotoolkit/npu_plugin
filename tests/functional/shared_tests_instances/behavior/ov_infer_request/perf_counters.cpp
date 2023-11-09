//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY),
                                               ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
                                              {ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY),
                                               ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<ov::AnyMap> autoConfigs = {{ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY),
                                              ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
                                             {ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY),
                                              ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

}  // namespace
