// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/skip_tests_config.hpp>
#include "behavior/core_integration.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common/functions.h"
#include "vpux_private_config.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassExecutableNetworkGetMetricTest_nightly = IEClassExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetConfigTest_nightly = IEClassExecutableNetworkGetConfigTest;

using IEClassGetMetricTest_nightly = IEClassGetMetricTest;
using IEClassGetConfigTest_nightly = IEClassGetConfigTest;

namespace {
std::vector<std::string> devices = {
    std::string(CommonTestUtils::DEVICE_KEEMBAY),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string("VPUXPlugin"), std::string(CommonTestUtils::DEVICE_KEEMBAY)),
};

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        IEClassBasicTestP_smoke, IEClassBasicTestP,
        ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassNetworkTestP_smoke, IEClassNetworkTestP,
        ::testing::ValuesIn(devices));

//
// IEClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using IEClassNetworkTestP_VPU = IEClassNetworkTestP;

TEST_P(IEClassNetworkTestP_VPU, smoke_ImportNetworkNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(IEClassNetworkTestP_VPU, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualNetwork, deviceName));
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(fileName));
    }
    if (CommonTestUtils::fileExists(fileName)) {
        {
            std::ifstream strm(fileName);
            SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    if (executableNetwork) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

using IEClassNetworkTestP_VPU_GetMetric = IEClassNetworkTestP_VPU;

TEST_P(IEClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    Core ie;
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 1);
    ASSERT_EQ(optimizationCapabilities.front(), METRIC_VALUE(FP16));
}

INSTANTIATE_TEST_CASE_P(
        DISABLED_smoke_IEClassGetMetricP, IEClassNetworkTestP_VPU_GetMetric,
        ::testing::ValuesIn(devices));

// TODO: enable with HETERO
INSTANTIATE_TEST_CASE_P(
        DISABLED_smoke_IEClassImportExportTestP, IEClassNetworkTestP_VPU,
        ::testing::Values(std::string(CommonTestUtils::DEVICE_KEEMBAY)));

#if defined(ENABLE_MKL_DNN) && ENABLE_MKL_DNN

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP_HETERO_CPU, IEClassNetworkTestP_VPU,
        ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY) + ",CPU"));

#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        DISABLED_DISABLED_IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassExecutableNetworkGetConfigTest_nightly,
        IEClassExecutableNetworkGetConfigTest,
        ::testing::ValuesIn(devices));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassGetConfigTest_nightly,
        IEClassGetConfigTest,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetConfigTest_nightly,
        IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

// IE Class Query network

class IEClassQueryNetworkTest_VPU : public IEClassQueryNetworkTest {
public:
    void SetUp() override {
       IEClassQueryNetworkTest::SetUp();
       config[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
    }
protected:
    std::map<std::string, std::string> config;
};

TEST_P(IEClassQueryNetworkTest_VPU, QueryNetworkWithCorrectDeviceID) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleNetwork, deviceName + "." + deviceIDs[0], config));
    } else {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_CASE_P(
        IEClassQueryNetworkTest_smoke,
        IEClassQueryNetworkTest_VPU,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassQueryNetworkTest_smoke,
        IEClassQueryNetworkTest,
        ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
        IEClassLoadNetworkTest_smoke,
        IEClassLoadNetworkTest,
        ::testing::ValuesIn(devices));

TEST_P(IEClassLoadNetworkTest, checkBlobCachingSingleDevice) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    CommonTestUtils::removeFilesWithExt("cache", "blob");
    CommonTestUtils::removeDir("cache");
    Core ie;
    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "cache/"}});

    if (supportsDeviceID(ie, deviceName)) {
        const auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP() << "No devices available";

        // ***********************************************
        // TODO Get rid of this skip - TBH is detected as KMB B0 (swID by XLink is incorrect)
        const auto numDev3700 = std::count_if(deviceIDs.cbegin(), deviceIDs.cend(), [](const std::string& devName) -> bool {
            return (devName.find("3700.") == 0);
        });
        if (numDev3700 > 1) {
            GTEST_SKIP() << "TBH incorrect swID";
        }
        // ***********************************************

        std::string architecture;
        const std::string fullDeviceName = deviceName + "." + deviceIDs[0];
        ASSERT_NO_THROW(architecture = ie.GetMetric(fullDeviceName, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>());

        auto start_time = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleNetwork, fullDeviceName));
        std::chrono::duration<double> first_time = std::chrono::steady_clock::now() - start_time;
        start_time = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleNetwork, fullDeviceName));
        std::chrono::duration<double> second_time = std::chrono::steady_clock::now() - start_time;

        std::cout << "[TIME] First LoadNetwork time: " << first_time.count() << std::endl;
        std::cout << "[TIME] Second LoadNetwork time: " << second_time.count() << std::endl;

        CommonTestUtils::removeFilesWithExt("cache", "blob");
        CommonTestUtils::removeDir("cache");

        ASSERT_GE(first_time.count(), second_time.count());
    } else {
        GTEST_SKIP() << "No support deviceID";
    }
}

//
// IE Class GetMetric
//

using IEClassGetMetricTest_DEVICE_ARCHITECTURE = IEClassBaseTestP;

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetMetricAndPrint) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;
    const auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_SKIP() << "No devices available";

    ASSERT_METRIC_SUPPORTED(METRIC_KEY(DEVICE_ARCHITECTURE));

    std::string architecture;
    ASSERT_NO_THROW(architecture = ie.GetMetric(deviceName + "." + deviceIDs[0], METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>());

    std::cout << "Architect type: " << architecture << std::endl;
}

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetMetricWithoutDevice) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    if (supportsDeviceID(ie, deviceName)) {
        const auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (!deviceIDs.empty())
            GTEST_SKIP() << "Devices list not empty";

        ASSERT_ANY_THROW(ie.GetMetric(deviceName, METRIC_KEY(DEVICE_ARCHITECTURE)));
    }
}

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetAllArchitectures) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Core ie;

    const auto deviceIDs = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_SKIP() << "No devices available";
    for (const auto& item : deviceIDs) {
        std::string architecture;
        ASSERT_NO_THROW(architecture = ie.GetMetric(deviceName + "." + item, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>());
        std::cout << "Architect type: " << architecture << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_DEVICE_ARCHITECTURE,
        ::testing::ValuesIn(devices));

} // namespace
