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
#include "vpux_metrics.h"
#include "vpux_private_config.hpp"

namespace vpux {

Metrics::Metrics(const VPUXBackends::CPtr& backends): _backends(backends) {
    _supportedMetrics = {
            METRIC_KEY(SUPPORTED_METRICS),         METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(FULL_DEVICE_NAME),          METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES), METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
            METRIC_KEY(RANGE_FOR_STREAMS),         METRIC_KEY(IMPORT_EXPORT_SUPPORT),
    };

    _supportedConfigKeys = {
            CONFIG_KEY(LOG_LEVEL),
            CONFIG_KEY(PERF_COUNT),
            CONFIG_KEY(DEVICE_ID),
            VPUX_CONFIG_KEY(THROUGHPUT_STREAMS),
            KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
            VPUX_CONFIG_KEY(PLATFORM),
    };
}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backends == nullptr ? std::vector<std::string>() : _backends->getAvailableDevicesNames();
}

// TODO each backend may support different metrics
const std::vector<std::string>& Metrics::SupportedMetrics() const {
    return _supportedMetrics;
}

// TODO: Need to add the full name
std::string Metrics::GetFullDevicesNames() const {
    return {""};
}

// TODO each backend may support different configs
const std::vector<std::string>& Metrics::GetSupportedConfigKeys() const {
    return _supportedConfigKeys;
}

// TODO each backend may support different optimization capabilities
const std::vector<std::string>& Metrics::GetOptimizationCapabilities() const {
    return _optimizationCapabilities;
}

const std::tuple<uint32_t, uint32_t, uint32_t>& Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& Metrics::GetRangeForStreams() const {
    return _rangeForStreams;
}

}  // namespace vpux
