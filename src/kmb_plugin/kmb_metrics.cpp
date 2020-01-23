// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_metrics.h"

#include <algorithm>
#include <vpu/utils/error.hpp>

using namespace vpu::KmbPlugin;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::PluginConfigParams;

//------------------------------------------------------------------------------
// Implementation of methods of class KmbMetrics
//------------------------------------------------------------------------------

KmbMetrics::KmbMetrics() {
    _supportedMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(AVAILABLE_DEVICES),
    };
}

std::vector<std::string> KmbMetrics::AvailableDevicesNames() const {
    // TODO replace with xlink_get_device_list filtered via xlink_get_device_status when API becomes available
    std::vector<std::string> availableDevices = {"Gen3 Intel(R) Movidius(TM) VPU code-named Keem Bay"};

    std::sort(availableDevices.begin(), availableDevices.end());
    return availableDevices;
}

const std::vector<std::string>& KmbMetrics::SupportedMetrics() const { return _supportedMetrics; }
