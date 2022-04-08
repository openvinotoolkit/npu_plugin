// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"
#include <functional_test_utils/skip_tests_config.hpp>
#include "common/functions.h"
#include "common_test_utils/file_utils.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_metrics.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassGetMetricTest_nightly = IEClassGetMetricTest;
using IEClassGetConfigTest_nightly = IEClassGetConfigTest;

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

INSTANTIATE_TEST_SUITE_P(IEClassBasicTestP_smoke, IEClassBasicTestP, ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(DISABLED_IEClassNetworkTestP_smoke, IEClassNetworkTestP, ::testing::ValuesIn(devices));

//
// IEClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using IEClassNetworkTestP_VPU = IEClassNetworkTestP;

TEST_P(IEClassNetworkTestP_VPU, smoke_ImportNetworkNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    std::stringstream strm;
    InferenceEngine::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, target_device));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, target_device));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(IEClassNetworkTestP_VPU, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, target_device));
        SKIP_IF_NOT_IMPLEMENTED(executableNetwork.Export(fileName));
    }
    if (CommonTestUtils::fileExists(fileName)) {
        {
            std::ifstream strm(fileName);
            SKIP_IF_NOT_IMPLEMENTED(executableNetwork = ie.ImportNetwork(strm, target_device));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    if (executableNetwork) {
        ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    }
}

using IEClassNetworkTestP_VPU_GetMetric = IEClassNetworkTestP_VPU;

TEST_P(IEClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    InferenceEngine::Core ie;
    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    InferenceEngine::Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter =
                            ie.GetMetric(target_device, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 1);
    ASSERT_EQ(optimizationCapabilities.front(), METRIC_VALUE(FP16));
}

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_IEClassGetMetricP, IEClassNetworkTestP_VPU_GetMetric,
                         ::testing::ValuesIn(devices));

// TODO: enable with HETERO
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_IEClassImportExportTestP, IEClassNetworkTestP_VPU,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_KEEMBAY)));

#if defined(ENABLE_MKL_DNN) && ENABLE_MKL_DNN

INSTANTIATE_TEST_SUITE_P(smoke_IEClassImportExportTestP_HETERO_CPU, IEClassNetworkTestP_VPU,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_KEEMBAY) + ",CPU"));

#endif

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_AVAILABLE_DEVICES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(DISABLED_IEClassGetConfigTest_nightly, IEClassGetConfigTest, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(IEClassGetConfigTest_nightly, IEClassGetConfigTest_ThrowUnsupported,
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
    InferenceEngine::Core ie;

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP();
        ASSERT_NO_THROW(ie.QueryNetwork(simpleCnnNetwork, target_device + "." + deviceIDs[0], config));
    } else {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_SUITE_P(IEClassQueryNetworkTest_smoke, IEClassQueryNetworkTest_VPU, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(DISABLED_IEClassQueryNetworkTest_smoke, IEClassQueryNetworkTest, ::testing::ValuesIn(devices));

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_IEClassLoadNetworkTest_smoke, IEClassLoadNetworkTest,
                         ::testing::ValuesIn(devices));

TEST_P(IEClassLoadNetworkTest, checkBlobCachingSingleDevice) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    if (PlatformEnvironment::PLATFORM.find("_EMU") != std::string::npos)
        GTEST_SKIP() << "Test disabled for emulator platform.";

    CommonTestUtils::removeFilesWithExt("cache", "blob");
    CommonTestUtils::removeDir("cache");
    InferenceEngine::Core ie;

    // [Track number: E#20961]
    if (getBackendName(ie) == "LEVEL0") {
        GTEST_SKIP() << "Sporadic failures on Level0 - bad results.";
    }

    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "cache/"}});

    if (supportsDeviceID(ie, target_device)) {
        const auto deviceIDs =
                ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP() << "No devices available";

        // ***********************************************
        // TODO Get rid of this skip - VPU311X is detected as KMB B0 (swID by XLink is incorrect)
        const auto numDev3700 =
                std::count_if(deviceIDs.cbegin(), deviceIDs.cend(), [](const std::string& devName) -> bool {
                    return (devName.find("3700.") == 0);
                });
        if (numDev3700 > 1) {
            GTEST_SKIP() << "VPU311X incorrect swID";
        }
        // ***********************************************

        std::string architecture;
        const std::string fullDeviceName = target_device + "." + deviceIDs[0];
        ASSERT_NO_THROW(architecture = ie.GetMetric(fullDeviceName, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>());

        auto start_time = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleCnnNetwork, fullDeviceName));
        std::chrono::duration<double> first_time = std::chrono::steady_clock::now() - start_time;
        start_time = std::chrono::steady_clock::now();
        ASSERT_NO_THROW(ie.LoadNetwork(simpleCnnNetwork, fullDeviceName));
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

using IEClassGetMetricTest_DEVICE_ARCHITECTURE = BehaviorTestsUtils::IEClassBaseTestP;

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetMetricAndPrint) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_SKIP() << "No devices available";

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(DEVICE_ARCHITECTURE));

    std::string architecture;
    ASSERT_NO_THROW(architecture = ie.GetMetric(target_device + "." + deviceIDs[0], METRIC_KEY(DEVICE_ARCHITECTURE))
                                           .as<std::string>());

    std::cout << "Architect type: " << architecture << std::endl;
}

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetMetricWithoutDeviceThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;

    if (supportsDeviceID(ie, target_device)) {
        const auto deviceIDs =
                ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (!deviceIDs.empty())
            GTEST_SKIP() << "Devices list is not empty";

        ASSERT_ANY_THROW(ie.GetMetric(target_device, METRIC_KEY(DEVICE_ARCHITECTURE)));
    }
}

TEST_P(IEClassGetMetricTest_DEVICE_ARCHITECTURE, GetAllArchitectures) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;

    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty())
        GTEST_SKIP() << "No devices available";
    for (const auto& item : deviceIDs) {
        std::string architecture;
        ASSERT_NO_THROW(
                architecture =
                        ie.GetMetric(target_device + "." + item, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>());
        std::cout << "Architect type: " << architecture << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_DEVICE_ARCHITECTURE,
                        ::testing::ValuesIn(devices));

// Testing private VPUX Plugin metric "BACKEND_NAME"
using IEClassGetMetricTest_BACKEND_NAME = BehaviorTestsUtils::IEClassBaseTestP;

TEST_P(IEClassGetMetricTest_BACKEND_NAME, GetBackendName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;

    const std::unordered_set<std::string> availableBackends = {"LEVEL0", "EMULATOR"};
    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    const auto backendName = ie.GetMetric(target_device, VPUX_METRIC_KEY(BACKEND_NAME)).as<std::string>();
    std::cout << "Devices: " << deviceIDs.size() << std::endl;
    std::cout << "Backend name: " << backendName << std::endl;
    if (deviceIDs.empty()) {
        ASSERT_TRUE(backendName.empty());
    } else {
        ASSERT_TRUE(availableBackends.find(backendName) != availableBackends.end());
    }
}

INSTANTIATE_TEST_CASE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_BACKEND_NAME, ::testing::ValuesIn(devices));

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricForAllDevicesWithDeviceID) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty()) {
        GTEST_SKIP() << "No devices available";
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(FULL_DEVICE_NAME));
    std::string fullDevName;
    for (const auto& deviceID : deviceIDs) {
        ASSERT_NO_THROW(
                fullDevName =
                        ie.GetMetric(target_device + "." + deviceID, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>());
        ASSERT_FALSE(fullDevName.empty());
        std::cout << "Full device name: " << fullDevName << std::endl;
    }
}

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricForSingleDeviceWithoutDeviceID) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.size() != 1) {
        GTEST_SKIP() << "Not single device case";
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(FULL_DEVICE_NAME));
    std::string fullDevName;
    ASSERT_NO_THROW(fullDevName = ie.GetMetric(target_device, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>());
    ASSERT_FALSE(fullDevName.empty());
    std::cout << "Full device name: " << fullDevName << std::endl;
}

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricWithoutDeviceThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;

    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (!deviceIDs.empty()) {
        GTEST_SKIP() << "Devices list is not empty";
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(FULL_DEVICE_NAME));
    ASSERT_ANY_THROW(ie.GetMetric(target_device, METRIC_KEY(FULL_DEVICE_NAME)));
}

TEST_P(IEClassGetMetricTest_FULL_DEVICE_NAME, GetMetricWithIncorrectDeviceIDThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferenceEngine::Core ie;
    const auto deviceIDs = ie.GetMetric(target_device, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
    if (deviceIDs.empty()) {
        GTEST_SKIP() << "No devices available";
    }

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(FULL_DEVICE_NAME));
    std::vector<std::string> incorrectDevIds = {"IncorrectDevId", "Platform.slice", "3000.0"};
    for (const auto& incorrectDevId : incorrectDevIds) {
        ASSERT_ANY_THROW(
                ie.GetMetric(target_device + "." + incorrectDevId, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>());
    }
}

INSTANTIATE_TEST_CASE_P(IEClassGetMetricTest_nightly, IEClassGetMetricTest_FULL_DEVICE_NAME,
                        ::testing::ValuesIn(devices));

}  // namespace
