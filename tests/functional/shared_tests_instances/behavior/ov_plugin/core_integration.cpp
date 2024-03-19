//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "overload/ov_plugin/core_integration.hpp"
#include "vpux/properties.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"

using namespace ov::test::behavior;
using namespace LayerTestsUtils;

namespace {
std::vector<std::string> devices = {
        std::string(ov::test::utils::DEVICE_NPU),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string(vpux::VPUX_PLUGIN_LIB_NAME), std::string(ov::test::utils::DEVICE_NPU)),
};

namespace OVClassBasicTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::pair<std::string, std::string>> obj) {
    std::ostringstream result;
    result << "OVClassBasicTestName_" << obj.param.first << "_" << obj.param.second;
    result << "_targetDevice=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassBasicTestName

namespace OVClassNetworkTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    result << "OVClassNetworkTestName_" << obj.param;
    result << "_targetDevice=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassNetworkTestName

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassBasicTestP, OVClassBasicTestP, ::testing::ValuesIn(plugins),
                         OVClassBasicTestName::getTestCaseName);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(smoke_OVClassBasicTestP, OVClassBasicTestPVpux, ::testing::ValuesIn(plugins),
                         OVClassBasicTestName::getTestCaseName);
#endif

INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP, OVClassNetworkTestPVpux, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

//
// OVClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP, OVClassImportExportTestP,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_NPU),
                                           "HETERO:" + std::string(ov::test::utils::DEVICE_NPU)),
                         OVClassNetworkTestName::getTestCaseName);

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU, OVClassImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(ov::test::utils::DEVICE_NPU) + ",CPU"));
#endif

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_AVAILABLE_DEVICES,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly, OVClassGetMetricTest_DEVICE_UUID, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetLogLevelConfigTest, OVClassSetLogLevelConfigTest,
                         ::testing::Values("MULTI", "AUTO"), OVClassNetworkTestName::getTestCaseName);

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU)}};
const std::vector<ov::AnyMap> configsDeviceProperties = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU, ov::num_streams(4))}};
const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigTest, OVClassSetDevicePriorityConfigTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"), ::testing::ValuesIn(multiConfigs)));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_VPUX_OVClassLoadNetworkWithCorrectSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
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
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTestVpux, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

//
// VPU specific metrics
//

using OVClassGetMetricAndPrintNoThrow = OVClassBaseTestP;
TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeLesserThanTotalMemSize) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_vpux::device_total_mem_size.name()));
    uint64_t t = p.as<uint64_t>();
    ASSERT_NE(t, 0);

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_vpux::device_alloc_mem_size.name()));
    uint64_t a = p.as<uint64_t>();

    ASSERT_LT(a, t);

    std::cout << "OV NPU device alloc/total memory size: " << a << "/" << t << std::endl;
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeLesserAfterModelIsLoaded) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_vpux::device_alloc_mem_size.name()));
    uint64_t a1 = p.as<uint64_t>();

    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto model = ngraph::builder::subgraph::makeConvPoolRelu();
        ASSERT_NO_THROW(ie.compile_model(model, target_device));
    }

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_vpux::device_alloc_mem_size.name()));
    uint64_t a2 = p.as<uint64_t>();

    std::cout << "OV NPU device {alloc before load network/alloc after load network} memory size: {" << a1 << "/" << a2
              << "}" << std::endl;

    // after the network is loaded onto device, allocated memory value should increase
    ASSERT_LE(a1, a2);
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDriverVersion) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_vpux::driver_version.name()));
    uint32_t t = p.as<uint32_t>();

    std::cout << "NPU driver version is " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_vpux::driver_version.name());
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricAndPrintNoThrow,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), OVClassNetworkTestName::getTestCaseName);

}  // namespace
