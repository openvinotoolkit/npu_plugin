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

#include <functional>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "hddl2_params.hpp"

using RangeType = std::tuple<unsigned int, unsigned int, unsigned int>;

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
// class HDDL2Metrics
// Class to keep and extract metrics value.
//------------------------------------------------------------------------------

class HDDL2Metrics {
public:
    // Constructor
    HDDL2Metrics();

    // Accessors
    static std::vector<std::string> GetAvailableExecutionCoresNames();
    static std::vector<std::string> GetAvailableDeviceNames();
    static bool isAvailableDevices();
    const std::vector<std::string>& SupportedMetrics() const;

    // Destructor
    ~HDDL2Metrics() = default;

private:
    // Data section
    std::vector<std::string> _supportedMetrics;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
