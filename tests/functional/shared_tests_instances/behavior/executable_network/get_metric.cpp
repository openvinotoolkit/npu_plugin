// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/executable_network/get_metric.hpp"
#include <functional_test_utils/skip_tests_config.hpp>
#include "common/functions.h"
#include "common/utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "vpux/properties.hpp"
#include "vpux_private_properties.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassExecutableNetworkGetMetricTest_nightly = IEClassExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetConfigTest_nightly = IEClassExecutableNetworkGetConfigTest;

namespace {
std::vector<std::string> devices = {
        std::string(ov::test::utils::DEVICE_NPU),
};

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetMetricTest_nightly,
                         IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetMetricTest_nightly,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetMetricTest_nightly,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetMetricTest_nightly,
                         IEClassExecutableNetworkGetMetricTest_NETWORK_NAME, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetMetricTest_nightly,
                         IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_SUITE_P(IEClassExecutableNetworkGetConfigTest_nightly, IEClassExecutableNetworkGetConfigTest,
                         ::testing::ValuesIn(devices));
}  // namespace
