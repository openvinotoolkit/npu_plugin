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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "kmb_executor.h"

using RangeType = std::tuple<unsigned int, unsigned int, unsigned int>;

namespace vpu {
namespace KmbPlugin {

//------------------------------------------------------------------------------
// class KmbMetrics
// Class to keep and extract metrics value.
//------------------------------------------------------------------------------

class KmbMetrics {
public:
    // Constructor
    KmbMetrics();

    // Accessors
    std::vector<std::string> AvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;
    static std::string GetFullDevicesNames();
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    const std::vector<std::string>& GetOptimizationCapabilities() const;
    const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() const;
    const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() const;

    // Destructor
    ~KmbMetrics() = default;

private:
    // Data section
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    std::vector<std::string> _optimizationCapabilities;
    std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests;
    std::tuple<uint32_t, uint32_t> _rangeForStreams;
};

}  // namespace KmbPlugin
}  // namespace vpu
