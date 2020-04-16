//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpu_layers_tests.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

using GetMetricTest = vpuLayersTests;

TEST_F(GetMetricTest, getAvailableDevices) {
    Core ie;

    std::vector<std::string> kmbSupportedMetrics = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
    std::vector<std::string>::const_iterator kmbAvalableDevMetricIter =
        std::find(kmbSupportedMetrics.begin(), kmbSupportedMetrics.end(), METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_NE(kmbAvalableDevMetricIter, kmbSupportedMetrics.end());

    std::vector<std::string> kmbDeviceIds = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_FALSE(kmbDeviceIds.empty());
    ASSERT_EQ(kmbDeviceIds.size(), 1);
    ASSERT_NE(kmbDeviceIds.begin()->find("Keem Bay"), std::string::npos);

    std::cout << "Found available KMB devices: " << std::endl;
    for (const std::string& deviceId : kmbDeviceIds) {
        std::cout << deviceId << std::endl;
    }
}
