//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "vpu_test_env_cfg.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"
#include "vpux/vpux_metrics.hpp"

using namespace ov::test::behavior;
using namespace LayerTestsUtils;

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

INSTANTIATE_TEST_SUITE_P(smoke_OVClassBasicTestP, OVClassBasicTestP, ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::ValuesIn(devices));

//
// OVClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP, OVClassImportExportTestP,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_KEEMBAY),
                                           "HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU, OVClassImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY) + ",CPU"));
#endif

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_AVAILABLE_DEVICES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_DEVICE_UUID, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetLogLevelConfigTest, OVClassSetLogLevelConfigTest,
                         ::testing::Values("MULTI", "AUTO"));

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY)}};
const std::vector<ov::AnyMap> configsDeviceProperties = {
        {ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY, ov::num_streams(4))}};
const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
        {ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::device::priorities(CommonTestUtils::DEVICE_KEEMBAY),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(CommonTestUtils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(CommonTestUtils::DEVICE_KEEMBAY,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigTest, OVClassSetDevicePriorityConfigTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"), ::testing::ValuesIn(multiConfigs)));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest, ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_VPUX_OVClassLoadNetworkWithCorrectSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassLoadNetworkWithSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassLoadNetworkWithSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

// IE Class load and check network with ov::device::properties
// OVClassLoadNetworkAndCheckSecondaryPropertiesTest only works with property num_streams of type int32_t
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_VPUX_OVClassLoadNetworkAndCheckWithSecondaryPropertiesTest,
                         OVClassLoadNetworkAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::ValuesIn(devices));

//
// VPU specific metrics
//
static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    result << "targetDevice=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(obj.param);

    return result.str();
}

using OVClassGetMetricAndPrintNoThrow = OVClassBaseTestP;
TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceTotalMemSize) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, VPUX_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    uint64_t t = p.as<uint64_t>();

    std::cout << "OV NPU device total memory size: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(VPUX_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDriverVersion) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, VPUX_METRIC_KEY(DRIVER_VERSION)));
    uint32_t t = p.as<uint32_t>();

    std::cout << "NPU driver version is " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(VPUX_METRIC_KEY(DRIVER_VERSION));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricAndPrintNoThrow,
                         ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY), getTestCaseName);

}  // namespace
