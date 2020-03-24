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

#include <RemoteMemory.h>
#include <WorkloadContext.h>
#include <details/ie_cnn_network_tools.h>

#include <blob_factory.hpp>
#include <fstream>

#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"
#include "ie_plugin_config.hpp"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

using Metrics_Tests = ::testing::Test;

TEST_F(Metrics_Tests, getAvailableDevices) {
    std::string deviceName = "HDDL2";
    IE::Core ie;
    std::vector<std::string> hddl2SupportedMetrics = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));

    std::vector<std::string>::const_iterator hddl2AvalableDevMetricIter =
        std::find(hddl2SupportedMetrics.begin(), hddl2SupportedMetrics.end(), METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_NE(hddl2AvalableDevMetricIter, hddl2SupportedMetrics.end());

    std::vector<std::string>::const_iterator hddl2AvalableExeCoreMetricIter = std::find(
        hddl2SupportedMetrics.begin(), hddl2SupportedMetrics.end(), VPU_HDDL2_METRIC(GET_AVAILABLE_EXECUTION_CORES));
    ASSERT_NE(hddl2AvalableExeCoreMetricIter, hddl2SupportedMetrics.end());

    std::vector<std::string> hddl2DeviceIds = ie.GetMetric(deviceName, METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_FALSE(hddl2DeviceIds.empty());
    ASSERT_NE(hddl2DeviceIds.begin()->find("HDDL2Device"), std::string::npos);

    std::cout << "Found available HDDL2 devices: " << std::endl;
    for (const std::string& deviceId : hddl2DeviceIds) {
        std::cout << deviceId << std::endl;
    }

    std::vector<std::string> hddl2Cores = ie.GetMetric(deviceName, VPU_HDDL2_METRIC(GET_AVAILABLE_EXECUTION_CORES));
    ASSERT_FALSE(hddl2Cores.empty());

    std::cout << "Found available HDDL2 execution cores: " << std::endl;
    for (const std::string& core : hddl2Cores) {
        std::cout << core << std::endl;
    }
}
