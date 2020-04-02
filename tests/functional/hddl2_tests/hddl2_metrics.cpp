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

#include <core_api.h>

#include "HddlUnite.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"
#include "ie_metric_helpers.hpp"

namespace IE = InferenceEngine;

class Metrics_Tests : public CoreAPI_Tests {
public:
    std::vector<std::string> getHddlDevicesIds();
    std::vector<std::string> hddlDevicesIds;

protected:
    void SetUp() override;
};

std::vector<std::string> Metrics_Tests::getHddlDevicesIds() {
    std::vector<HddlUnite::Device> devices;
    std::vector<std::string> devicesNames;
    auto status = getAvailableDevices(devices);
    if (status == HDDL_OK) {
        for (const auto& device : devices) {
            devicesNames.push_back(std::to_string(device.getSwDeviceId()));
        }
    }
    return devicesNames;
}

void Metrics_Tests::SetUp() { hddlDevicesIds = getHddlDevicesIds(); }

TEST_F(Metrics_Tests, supportsGetAvailableDevice) {
    auto metricName = METRIC_KEY(AVAILABLE_DEVICES);

    std::vector<std::string> supportedMetrics = ie.GetMetric(pluginName, METRIC_KEY(SUPPORTED_METRICS));
    auto found_metric = std::find(supportedMetrics.begin(), supportedMetrics.end(), metricName);
    ASSERT_NE(found_metric, supportedMetrics.end());
}

TEST_F(Metrics_Tests, canGetAvailableDevice) {
    std::vector<std::string> availableHDDL2Devices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));

    ASSERT_GE(availableHDDL2Devices.size(), 1);
    for (const auto& id : hddlDevicesIds) {
        auto found_name =
            std::find_if(availableHDDL2Devices.begin(), availableHDDL2Devices.end(), [id](const std::string& str) {
                return str.find(id) != std::string::npos;
            });
        ASSERT_NE(found_name, availableHDDL2Devices.end());
    }
}

TEST_F(Metrics_Tests, canFoundHddl2DeviceInAllDevices) {
    std::vector<std::string> allDevices = ie.GetAvailableDevices();
    auto found_name = std::find_if(allDevices.begin(), allDevices.end(), [this](const std::string& str) {
        return str.find(pluginName) != std::string::npos;
    });
    ASSERT_NE(found_name, allDevices.end());
}

TEST_F(Metrics_Tests, canFoundHddl2DevicesIdsInAllDevices_IfMany) {
    if (hddlDevicesIds.size() <= 1) {
        GTEST_SKIP() << "Not enough devices for test";
    }
    std::vector<std::string> allDevices = ie.GetAvailableDevices();
    for (const auto& id : hddlDevicesIds) {
        auto found_name = std::find_if(allDevices.begin(), allDevices.end(), [id](const std::string& str) {
            return str.find(id) != std::string::npos;
        });
        ASSERT_NE(found_name, allDevices.end());
    }
}
