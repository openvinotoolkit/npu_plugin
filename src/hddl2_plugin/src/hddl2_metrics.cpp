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

// IE
#include <ie_metric_helpers.hpp>
// Plugin
#include "hddl2_exceptions.h"
#include "hddl2_metrics.h"

using namespace vpu::HDDL2Plugin;

HDDL2Metrics::HDDL2Metrics(const vpux::VPUXBackends::CPtr& backends): _backends(backends) {
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
}

std::vector<std::string> HDDL2Metrics::GetAvailableDevicesNames() const {
    return _backends->getAvailableDevicesNames();
}

// TODO each backend may support different metrics
const std::vector<std::string>& HDDL2Metrics::SupportedMetrics() const { return _supportedMetrics; }

// TODO: Need to add the full name
std::string HDDL2Metrics::GetFullDevicesNames() const { return {""}; }

// TODO each backend may support different configs
const std::vector<std::string>& HDDL2Metrics::GetSupportedConfigKeys() const { return _supportedConfigKeys; }

// TODO each backend may support different optimization capabilities
const std::vector<std::string>& HDDL2Metrics::GetOptimizationCapabilities() const { return _optimizationCapabilities; }

const std::tuple<uint32_t, uint32_t, uint32_t>& HDDL2Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& HDDL2Metrics::GetRangeForStreams() const { return _rangeForStreams; }
