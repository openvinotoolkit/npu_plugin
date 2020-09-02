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

#include "kmb_metrics.h"

#include <algorithm>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/error.hpp>
#include <vpux.hpp>

using namespace vpu::KmbPlugin;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::PluginConfigParams;

//------------------------------------------------------------------------------
// Implementation of methods of class KmbMetrics
//------------------------------------------------------------------------------

KmbMetrics::KmbMetrics(const std::map<std::string, std::shared_ptr<vpux::Device>>& devices) {
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
        KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
#ifdef ENABLE_M2I
        VPU_KMB_CONFIG_KEY(USE_M2I),
#endif
    };

    for (const auto& elem : devices) {
        const auto& device = elem.second;
        _availableDevices.emplace_back(device->getName());
    }
}

std::vector<std::string> KmbMetrics::AvailableDevicesNames() const { return _availableDevices; }

const std::vector<std::string>& KmbMetrics::SupportedMetrics() const { return _supportedMetrics; }

std::string KmbMetrics::GetFullDevicesNames() { return {"Gen3 Intel(R) Movidius(TM) VPU code-named Keem Bay"}; }

const std::vector<std::string>& KmbMetrics::GetSupportedConfigKeys() const { return _supportedConfigKeys; }

const std::vector<std::string>& KmbMetrics::GetOptimizationCapabilities() const { return _optimizationCapabilities; }

const std::tuple<uint32_t, uint32_t, uint32_t>& KmbMetrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& KmbMetrics::GetRangeForStreams() const { return _rangeForStreams; }
