//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "common/functions.h"
#include "functional_test_utils/plugin_cache.hpp"

using namespace ov::test::behavior;

namespace {
std::vector<std::string> devices = {
        std::string(CommonTestUtils::DEVICE_KEEMBAY),
};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP, OVClassExecutableNetworkImportExportTestP,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_KEEMBAY),
                                           "HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU, OVClassExecutableNetworkImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_NETWORK_NAME, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetConfigTest_nightly, OVClassExecutableNetworkGetConfigTest,
                         ::testing::ValuesIn(devices));
}  // namespace
