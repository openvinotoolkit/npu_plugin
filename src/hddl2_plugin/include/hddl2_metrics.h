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

#pragma once

#include <HddlUnite.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "hddl2_params.hpp"
#include "vpu/kmb_plugin_config.hpp"

namespace vpu {
namespace HDDL2Plugin {

class HDDL2Metrics {
public:
    HDDL2Metrics();

    static std::vector<std::string> GetAvailableDevicesNames();
    const std::vector<std::string>& SupportedMetrics() const;
    static std::string GetFullDevicesNames();
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    const std::vector<std::string>& GetOptimizationCapabilities() const;
    const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() const;
    const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() const;

    static bool isServiceAvailable();

    ~HDDL2Metrics() = default;

private:
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    const std::vector<std::string> _optimizationCapabilities = {METRIC_VALUE(INT8)};

    // Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
    const std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests = {4, 10, 1};

    // Metric to provide information about a range for streams.(bottom bound, upper bound)
    const std::tuple<uint32_t, uint32_t> _rangeForStreams = {1, 4};
};

}  // namespace HDDL2Plugin
}  // namespace vpu
