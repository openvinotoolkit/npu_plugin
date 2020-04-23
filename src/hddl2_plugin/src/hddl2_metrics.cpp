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
#include <fstream>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>

#include "hddl2_params.hpp"
#include "vpu/kmb_plugin_config.hpp"

using namespace vpu::HDDL2Plugin;

HDDL2Metrics::HDDL2Metrics() {
    _supportedMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(AVAILABLE_DEVICES),
        METRIC_KEY(FULL_DEVICE_NAME),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMIZATION_CAPABILITIES),
        METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
        METRIC_KEY(RANGE_FOR_STREAMS),
    };

    _supportedConfigKeys = {
        VPU_KMB_CONFIG_KEY(PLATFORM),
        CONFIG_KEY(DEVICE_ID),
        CONFIG_KEY(LOG_LEVEL),
    };

    _optimizationCapabilities = {METRIC_VALUE(INT8)};

    _rangeForAsyncInferRequests = std::tuple<uint32_t, uint32_t, uint32_t>(3, 6, 1);

    _rangeForStreams = std::tuple<uint32_t, uint32_t>(1, 4);
}

std::vector<std::string> HDDL2Metrics::GetAvailableDevicesNames() {
    if (!HDDL2Metrics::isServiceAvailable()) {
        // return empty device list if service is not available
        return std::vector<std::string>();
    }

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

bool HDDL2Metrics::isServiceAvailable() {
    const std::ifstream defaultService("/opt/intel/hddlunite/bin/hddl_scheduler_service");

    const std::string specifiedServicePath =
        std::getenv("KMB_INSTALL_DIR") != nullptr ? std::getenv("KMB_INSTALL_DIR") : "";
    const std::ifstream specifiedService(specifiedServicePath + std::string("/bin/hddl_scheduler_service"));

    return specifiedService.good() || defaultService.good();
}

std::string HDDL2Metrics::GetFullDevicesNames() { return {"ARM Cortex-A53"}; }

const std::vector<std::string>& HDDL2Metrics::GetSupportedConfigKeys() const { return _supportedConfigKeys; }

const std::vector<std::string>& HDDL2Metrics::GetOptimizationCapabilities() const { return _optimizationCapabilities; }

const std::tuple<uint32_t, uint32_t, uint32_t>& HDDL2Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& HDDL2Metrics::GetRangeForStreams() const { return _rangeForStreams; }
