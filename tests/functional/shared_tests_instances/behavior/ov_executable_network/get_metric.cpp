//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "functional_test_utils/plugin_cache.hpp"

using namespace ov::test::behavior;

namespace {
std::vector<std::string> devices = {
        std::string(ov::test::utils::DEVICE_NPU),
};

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    result << "OVClassExecutableNetworkGetMetricTest";
    result << "_targetDevice=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP, OVClassExecutableNetworkImportExportTestP,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_NPU),
                                           "HETERO:" + std::string(ov::test::utils::DEVICE_NPU)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU, OVClassExecutableNetworkImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(ov::test::utils::DEVICE_NPU) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, ::testing::ValuesIn(devices),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, ::testing::ValuesIn(devices),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, ::testing::ValuesIn(devices),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_NETWORK_NAME, ::testing::ValuesIn(devices),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::ValuesIn(devices), getTestCaseName);

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetConfigTest_nightly, OVClassExecutableNetworkGetConfigTest,
                         ::testing::ValuesIn(devices), getTestCaseName);

}  // namespace
