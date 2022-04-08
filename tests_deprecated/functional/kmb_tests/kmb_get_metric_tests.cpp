//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpu_layers_tests.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

using GetMetricTest = vpuLayersTests;

TEST_F(GetMetricTest, getAvailableDevices) {
    // [Track number: S#34628]
    if (deviceName == "HDDL2" || (deviceName == "VPUX"))
        GTEST_SKIP() << "No x86 device on CI";
    std::vector<std::string> kmbSupportedMetrics = core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
    std::vector<std::string>::const_iterator kmbAvalableDevMetricIter =
            std::find(kmbSupportedMetrics.begin(), kmbSupportedMetrics.end(), METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_NE(kmbAvalableDevMetricIter, kmbSupportedMetrics.end());

    std::vector<std::string> kmbDeviceIds = core->GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    if (deviceName == "HDDL2") {
        ASSERT_FALSE(kmbDeviceIds.empty());
    } else {
        // x86 host must not find any available devices in this test
        ASSERT_TRUE(kmbDeviceIds.empty());
    }
}

TEST_F(GetMetricTest, supportMetrics) {
    // [Track number: S#34628]
    if (deviceName == "HDDL2" || deviceName == "VPUX")
        GTEST_SKIP() << "No x86 device on CI";
    std::vector<std::string> supportedMetrics = core->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
    for (auto& metric : supportedMetrics) {
        ASSERT_NO_THROW(core->GetMetric(deviceName, metric));
    }
}
