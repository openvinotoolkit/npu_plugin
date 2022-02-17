// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/plugin_cache.hpp"
#include "common/functions.h"
#include "behavior/ov_plugin/core_integration.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"

using namespace ov::test::behavior;

namespace {
std::vector<std::string> devices = {
        std::string(CommonTestUtils::DEVICE_KEEMBAY),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string(vpux::VPUX_PLUGIN_LIB_NAME), std::string(CommonTestUtils::DEVICE_KEEMBAY)),
};

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(OVClassBasicTestP_smoke, OVClassBasicTestP, ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(OVClassNetworkTestP_smoke, OVClassNetworkTestP, ::testing::ValuesIn(devices));

//
// OVClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP,
                         OVClassImportExportTestP,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_KEEMBAY),
                                           "HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU,
                         OVClassImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY) + ",CPU"));
#endif

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_AVAILABLE_DEVICES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly,
                         OVClassGetConfigTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(OVClassQueryNetworkTest_smoke, OVClassQueryNetworkTest, ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(OVClassLoadNetworkTest_smoke, OVClassLoadNetworkTest, ::testing::ValuesIn(devices));
}  // namespace
