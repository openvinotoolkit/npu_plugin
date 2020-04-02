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
#include <core_api.h>

#include "comparators.h"
#include "file_reader.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"
#include "ie_plugin_config.hpp"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

class Metrics_Tests : public CoreAPI_Tests {};

TEST_F(Metrics_Tests, supportsGetAvailableDevice) {
    auto metricName = METRIC_KEY(AVAILABLE_DEVICES);

    std::vector<std::string> supportedMetrics = ie.GetMetric(pluginName, METRIC_KEY(SUPPORTED_METRICS));
    auto found_metric = std::find(supportedMetrics.begin(), supportedMetrics.end(), metricName);
    ASSERT_NE(found_metric, supportedMetrics.end());
}

TEST_F(Metrics_Tests, canGetAvailableDevice) {
    std::vector<std::string> availableDevices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));

    ASSERT_EQ(availableDevices.size(), 1);
    ASSERT_EQ(availableDevices[0], pluginName);
}

TEST_F(Metrics_Tests, supportsGetAvailableExecutionCores) {
    auto metricName = VPU_HDDL2_METRIC(GET_AVAILABLE_EXECUTION_CORES);

    std::vector<std::string> supportedMetrics = ie.GetMetric(pluginName, METRIC_KEY(SUPPORTED_METRICS));
    auto found_metric = std::find(supportedMetrics.begin(), supportedMetrics.end(), metricName);
    ASSERT_NE(found_metric, supportedMetrics.end());
    auto devices = ie.GetAvailableDevices();
}

TEST_F(Metrics_Tests, canGetExecutionCores) {
    std::vector<std::string> availableDevices = ie.GetMetric(pluginName, VPU_HDDL2_METRIC(GET_AVAILABLE_EXECUTION_CORES));

    ASSERT_EQ(availableDevices.size(), 1);
    auto found_name = availableDevices[0].find(pluginName);
    ASSERT_NE(found_name, std::string::npos);
}
