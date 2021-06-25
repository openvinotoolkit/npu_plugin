//
// Copyright 2021 Intel Corporation.
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

#pragma once

// System
#include <string>
#include <unordered_set>
#include <vector>
#include <vpux.hpp>
// Plugin
#include "vpux_backends.h"
// TODO should not be here
#include <vpux/vpux_plugin_config.hpp>

namespace vpux {

class Metrics {
public:
    Metrics(const VPUXBackends::CPtr& backends);

    std::vector<std::string> GetAvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;
    std::string GetFullDevicesNames() const;
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    const std::vector<std::string>& GetOptimizationCapabilities() const;
    const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() const;
    const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() const;
    std::string GetDeviceArchitecture(const std::string& specifiedDeviceName) const;

    ~Metrics() = default;

private:
    const VPUXBackends::CPtr _backends;
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    const std::vector<std::string> _optimizationCapabilities = {METRIC_VALUE(INT8)};
    vpu::Logger _logger;

    // Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
    const std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests{4u, 10u, 1u};

    // Metric to provide information about a range for streams.(bottom bound, upper bound)
    const std::tuple<uint32_t, uint32_t> _rangeForStreams{1u, 4u};
};

}  // namespace vpux
