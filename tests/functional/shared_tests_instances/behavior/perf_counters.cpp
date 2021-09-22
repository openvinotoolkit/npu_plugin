// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/perf_counters.hpp"
#include "kmb_layer_test.hpp"

using namespace BehaviorTestsDefinitions;
const LayerTestsUtils::KmbTestEnvConfig envConfig;

namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {{"VPUX_PLATFORM", envConfig.IE_KMB_TESTS_PLATFORM}},
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PerfCountersTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                    ::testing::ValuesIn(configs)),
                            PerfCountersTest::getTestCaseName);

}  // namespace

