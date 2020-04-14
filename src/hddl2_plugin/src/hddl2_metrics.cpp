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

HDDL2Metrics::HDDL2Metrics() {
    _supportedMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(AVAILABLE_DEVICES),
    };
}

std::vector<std::string> HDDL2Metrics::GetAvailableDevicesNames() {
    std::vector<HddlUnite::Device> devices;
    auto status = getAvailableDevices(devices);
    if (status != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to get devices names!";
    }

    std::vector<std::string> devicesNames;
    for (const auto& device : devices) {
        devicesNames.push_back(std::to_string(device.getSwDeviceId()));
    }
    std::sort(devicesNames.begin(), devicesNames.end());
    return devicesNames;
}

const std::vector<std::string>& HDDL2Metrics::SupportedMetrics() const { return _supportedMetrics; }
