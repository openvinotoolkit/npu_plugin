// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/perf_counters.hpp"
#include "kmb_layer_test.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, PerfCountersTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                    ::testing::ValuesIn(configs)),
                            PerfCountersTest::getTestCaseName);

}  // namespace

