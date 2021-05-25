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

#include <core_api.h>

#include "HddlUnite.h"
#include "vpux/vpux_plugin_params.hpp"
#include "ie_core.hpp"
#include "ie_metric_helpers.hpp"
#include <device_helpers.hpp>

namespace IE = InferenceEngine;

class Metrics_Tests : public CoreAPI_Tests {
public:
    std::vector<uint32_t> getHddlDevicesIds();
    std::vector<uint32_t> hddlDevicesIds;

protected:
    void SetUp() override;
};

std::vector<std::uint32_t> Metrics_Tests::getHddlDevicesIds() {
    std::vector<HddlUnite::Device> devices;
    std::vector<std::uint32_t> devicesIds;
    auto status = getAvailableDevices(devices);
    if (status == HDDL_OK) {
        for (const auto& device : devices) {
            devicesIds.push_back(device.getSwDeviceId());
        }
    }
    return devicesIds;
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
                return str.find(utils::getDeviceNameBySwDeviceId(id)) != std::string::npos;
            });
        ASSERT_NE(found_name, availableHDDL2Devices.end());
    }
}

TEST_F(Metrics_Tests, supportMetrics) {
    std::vector<std::string> supportedMetrics = ie.GetMetric(pluginName, METRIC_KEY(SUPPORTED_METRICS));
    for (auto& metric : supportedMetrics) {
        ASSERT_NO_THROW(ie.GetMetric(pluginName, metric));
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
            return str.find(utils::getDeviceNameBySwDeviceId(id)) != std::string::npos;
        });
        ASSERT_NE(found_name, allDevices.end());
    }
}
