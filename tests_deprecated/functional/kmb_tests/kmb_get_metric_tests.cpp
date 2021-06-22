//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpu_layers_tests.hpp"

// xlink is used in this test to determine the number of available VPUs
#if defined(__arm__) || defined(__aarch64__)
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

using GetMetricTest = vpuLayersTests;

static size_t getVpuCount() {
    size_t vpuCount = 0;
#if defined(__arm__) || defined(__aarch64__)
    xlink_error initResult = xlink_initialize();
    if (initResult != X_LINK_SUCCESS) {
        return 0;
    }

    constexpr size_t MAX_DEVICES = 8;
    uint32_t devIds[MAX_DEVICES];
    uint32_t devCount = 0;
    xlink_error getDevResult = xlink_get_device_list(devIds, &devCount);
    if (getDevResult != X_LINK_SUCCESS) {
        return 0;
    }

    constexpr uint32_t INTERFACE_TYPE_SELECTOR = 0x7000000;
    for (size_t devIdx = 0; devIdx < devCount; devIdx++) {
        if ((devIds[devIdx] & INTERFACE_TYPE_SELECTOR) == 0) {
            vpuCount++;
        }
    }
#endif
    return vpuCount;
}

TEST_F(GetMetricTest, getAvailableDevices) {
    // [Track number: S#34628]
#if defined(__arm__) || defined(__aarch64__)
    bool runningOnARM = true;
#else
    bool runningOnARM = false;
#endif
    if (deviceName == "HDDL2" || (deviceName == "VPUX" && !runningOnARM)) GTEST_SKIP() << "No x86 device on CI";
    std::vector<std::string> kmbSupportedMetrics = core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
    std::vector<std::string>::const_iterator kmbAvalableDevMetricIter =
        std::find(kmbSupportedMetrics.begin(), kmbSupportedMetrics.end(), METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_NE(kmbAvalableDevMetricIter, kmbSupportedMetrics.end());

    std::vector<std::string> kmbDeviceIds = core->GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (runningOnARM) {
        ASSERT_FALSE(kmbDeviceIds.empty());
        ASSERT_EQ(kmbDeviceIds.size(), getVpuCount());

        std::cout << "Found available KMB devices: " << std::endl;
        for (const std::string& deviceId : kmbDeviceIds) {
            std::cout << deviceId << std::endl;
        }
    } else {
        if (deviceName == "HDDL2") {
            ASSERT_FALSE(kmbDeviceIds.empty());
        } else {
            // x86 host must not find any available devices in this test
            ASSERT_TRUE(kmbDeviceIds.empty());
        }
    }
}

TEST_F(GetMetricTest, supportMetrics) {
    // [Track number: S#34628]
    if (deviceName == "HDDL2" || deviceName == "VPUX") GTEST_SKIP() << "No x86 device on CI";
    std::vector<std::string> supportedMetrics = core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
    for (auto& metric : supportedMetrics) {
        ASSERT_NO_THROW(core->GetMetric(deviceName, metric));
    }
}
