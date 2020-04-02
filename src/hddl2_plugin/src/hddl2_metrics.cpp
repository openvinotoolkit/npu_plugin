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

#include "hddl2_metrics.h"

#include <algorithm>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>

#include "hddl2_params.hpp"

using namespace vpu::HDDL2Plugin;
const std::string HDDL2Metrics::_deviceName= "HDDL2";

HDDL2Metrics::HDDL2Metrics() {
    _supportedMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(AVAILABLE_DEVICES),
        VPU_HDDL2_METRIC(GET_AVAILABLE_EXECUTION_CORES),
    };
}

std::vector<std::string> HDDL2Metrics::GetAvailableExecutionCoresNames() {
    std::vector<HddlUnite::Device> Cores;
    getAvailableDevices(Cores);

    std::vector<std::string> availableDevices;
    for (auto& core : Cores) {
        availableDevices.push_back(_deviceName + "." + std::to_string(core.getSwDeviceId()));
    }
    std::sort(availableDevices.begin(), availableDevices.end());
    return availableDevices;
}

std::vector<std::string> HDDL2Metrics::GetAvailableDeviceNames() {
    return {HDDL2Metrics::isAnyDeviceAvailable() ? _deviceName: ""};
}

bool HDDL2Metrics::isAnyDeviceAvailable() {
    std::vector<HddlUnite::Device> devices;
    getAvailableDevices(devices);
    return !devices.empty();
}

const std::vector<std::string>& HDDL2Metrics::SupportedMetrics() const { return _supportedMetrics; }
